from __future__ import annotations

import copy
from typing import Any, List, Optional, Tuple, Union

import numpy as np  # type: ignore
import torch
from sklearn.cluster import KMeans  # type: ignore
from torch.utils.data import DataLoader, Dataset


class Loader(DataLoader):
    def __init__(self, dataset: DSet, **kwargs: Any) -> None:
        super(Loader, self).__init__(dataset, **kwargs)


class RegressionDataset(Dataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(RegressionDataset, self).__init__()
        self.x: torch.Tensor
        self.y: torch.Tensor
        self.og_y: torch.Tensor

        # these are the parameters of the normalization
        self.mu: torch.Tensor
        self.sigma: torch.Tensor
        self.y_mu: torch.Tensor
        self.y_sigma: torch.Tensor

        # the prototype of the data in this dataset
        self.prototype: torch.Tensor

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        return self.x[i], self.y[i], torch.tensor([i])

    def __len__(self) -> int:
        return self.x.size(0)

    def prune(self, idx: torch.Tensor) -> None:
        self.x = self.x[idx]
        self.y = self.y[idx]

    def get_y_moments(self) -> Tuple[float, float]:
        return self.y_mu.item(), self.y_sigma.item()

    def get_x_moments(self) -> Tuple[torch.Tensor, ...]:
        return self.mu, self.sigma

    def set_name(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def get_feature_ranges(self) -> torch.Tensor:
        """
        get the feature ranges for all of the x features, this was originally used
        to determine the ranges of features for generating adversarial examples as done by
        deep ensembles paper https://arxiv.org/abs/1612.01474
        """
        return torch.abs(self.x.min(dim=0)[0] - self.x.max(dim=0)[0])

    def valid(self) -> None:
        if torch.any(torch.isinf(self.x)) or torch.any(torch.isnan(self.x)):
            raise ValueError("x has invalid values")
        elif torch.any(torch.isinf(self.y)) or torch.any(torch.isnan(self.y)):
            raise ValueError("y has invalid values")

    def one_normalize(
        self, max_x: Optional[torch.Tensor] = None, max_y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """one normalize the dataset by dividing by the max value of x and y"""
        if max_x is None or max_y is None:
            max_x = torch.abs(self.x).max(dim=0)[0]
            max_y = torch.abs(self.y).max()

            self.max_x = max_x
            self.max_y = max_y

            # if maximum x value is zero, we want to avoid dividing by zero
            max_x[max_x == 0] = 1
            self.og_y = torch.clone(self.y)
            self.x /= max_x
            self.y /= max_y
            return max_x, max_y

        self.max_x = max_x
        self.max_y = max_y

        self.og_y = torch.clone(self.y)
        self.x /= max_x
        self.y /= max_y
        return max_x, max_y  # type: ignore

    def standard_normalize(
        self,
        mu: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        y_mu: Optional[torch.Tensor] = None,
        y_sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """standard normalize the dataset by ([x,y] - mu) / sigma"""
        if mu is None or sigma is None or y_mu is None or y_sigma is None:

            self.mu = self.x.mean(dim=0)
            self.sigma = self.x.std(dim=0)
            self.sigma[self.sigma == 0] = 1

            if torch.any(self.sigma == 0):
                raise ValueError(
                    "sigma should not have zero values, see what is going on here"
                )
                self.sigma[self.sigma == 0] = 1

            self.x = (self.x - self.mu) / self.sigma
            if hasattr(self, "prototype"):
                self.prototype = (self.prototype - self.mu) / self.sigma

            self.y_mu = self.y.mean()
            self.y_sigma = self.y.std()

            self.og_y = torch.clone(self.y)
            self.y = (self.y - self.y_mu) / self.y_sigma

            self.valid()

            return self.mu, self.sigma, self.y_mu, self.y_sigma

        self.mu = mu
        self.sigma = sigma
        self.y_mu = y_mu
        self.y_sigma = y_sigma

        self.x = (self.x - self.mu) / self.sigma
        if hasattr(self, "prototype"):
            self.prototype = (self.prototype - self.mu) / self.sigma

        self.og_y = torch.clone(self.y)
        self.y = (self.y - self.y_mu) / self.y_sigma

        self.valid()
        return mu, sigma, y_mu, y_sigma

    def sample(
        self, n: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """get a random sample of size n of the dataset"""
        perm = torch.randperm(self.x.size(0))
        return self.x[perm[:n]].to(device), self.y[perm[:n]].to(device), perm[:n]

    def get_test_cluster(
        self, n_clusters: int, tgt_test_size: float, kmeans: KMeans
    ) -> int:
        closest_cluster, closest_count = 0, 0
        for i in range(n_clusters):
            ith_cluster_count = (kmeans.labels_ == i).sum()
            if abs(ith_cluster_count - tgt_test_size) < closest_count:
                closest_count = ith_cluster_count
                closest_cluster = i

        return closest_cluster

    def get_cluster_indices(self, n_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """get cluster indices, returns indices of the train set and the test set"""
        kmeans = KMeans(n_clusters=n_clusters).fit(self.x.numpy())

        closest_cluster = self.get_test_cluster(n_clusters, 1.0 / n_clusters, kmeans)

        train_clusters = kmeans.labels_ != closest_cluster
        test_clusters = kmeans.labels_ == closest_cluster

        train_idx = np.argwhere(train_clusters).reshape(-1)
        test_idx = np.argwhere(test_clusters).reshape(-1)

        return torch.from_numpy(train_idx), torch.from_numpy(test_idx)

    def clustered_datasets_no_shift(
        self, n_clusters: int, batch_size: int, seed: int
    ) -> Tuple[List[DataLoader], ...]:
        """
        create n clusters of data for 'meta' datasets and return them as individual dataloaders
        if we want to use these as meta datasets later then they will need to be split further into individual
        training and test sets.

        This version of the clustered datasets does not contain a held out cluster for the test set, the test set is 
        composed of the same clusters as the training sets
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(self.x.numpy())

        # these will be the returned training and testing tasks
        train_sets: List[RegressionDataset] = []
        test_sets: List[RegressionDataset] = []

        # these are the training and testing indices which we will use for normalization
        train_idx: np.array = []
        test_idx: np.array = []

        # total ratio of instances which have been used for training
        train_inst = 0.0
        for label in range(n_clusters):
            cluster_idx = np.argwhere(kmeans.labels_ == label).squeeze().tolist()

            cluster_train_idx = cluster_idx[: int(len(cluster_idx) * 0.75)]
            cluster_test_idx = cluster_idx[int(len(cluster_idx) * 0.75) :]

            # make the training sets
            ds = RegressionDataset("placeholder")
            # confusing, but this sets the name of the new dataset to self so that it prints in the results files
            ds.set_name(self.__str__())
            ds.x = self.x[cluster_train_idx]
            ds.y = self.y[cluster_train_idx]
            ds.prototype = torch.from_numpy(kmeans.cluster_centers_[label])

            train_idx += cluster_train_idx
            train_sets.append(ds)
            train_inst += sum(cluster_train_idx)

            # make the testing sets
            ds = RegressionDataset("placeholder")
            # confusing, but this sets the name of the new dataset to self so that it prints in the results files
            ds.set_name(self.__str__())
            ds.x = self.x[cluster_test_idx]
            ds.y = self.y[cluster_test_idx]
            ds.prototype = torch.from_numpy(kmeans.cluster_centers_[label])

            test_idx += cluster_test_idx
            test_sets.append(ds)

        # hack to get the test set of the clustered meta set contain all of the examples.
        # This is because it was hard to change the stats object in all of the models training files to take a variable length
        # based on many datasets
        test_set = test_sets[0]
        for s in test_sets[1:]:
            test_set.x = torch.cat((test_set.x, s.x))
            test_set.y = torch.cat((test_set.y, s.y))

        test_sets = [test_set]

        # clone self and set the x and y to that of the training indices so we can standard
        # normalize everythhing based off of the training indices.
        dataset = copy.deepcopy(self)
        dataset.x = dataset.x[train_idx]
        dataset.y = dataset.y[train_idx]

        # normalize the training and the test set based off of the normalization parameters of training set
        x_mu, x_sigma, y_mu, y_sigma = dataset.standard_normalize()
        for s in train_sets:
            s.standard_normalize(x_mu, x_sigma, y_mu, y_sigma)

        for s in test_sets:
            s.standard_normalize(x_mu, x_sigma, y_mu, y_sigma)

        assert len(train_sets) > 0, "meta train sets should be more than 0"
        assert len(test_sets) > 0, "met test tasks should be greater than 0"

        return (
            [
                DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False,)
                for ds in train_sets
            ],
            [DataLoader(ds, batch_size=batch_size, num_workers=4) for ds in test_sets],
        )

    def clustered_datasets_shift(
        self, n_clusters: int, batch_size: int, seed: int
    ) -> Tuple[List[DataLoader], ...]:
        """
        create n clusters of data for 'meta' datasets and return them as individual dataloaders
        if we want to use these as meta datasets later then they will need to be split further into individual
        training and test sets.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(self.x.numpy())

        # these will be the returned training and testing tasks
        train_sets: List[RegressionDataset] = []
        test_sets: List[RegressionDataset] = []

        # these are the training and testing indices which we will use for normalization
        train_idx: np.array = []
        test_idx: np.array = []

        # total ratio of instances which have been used for training
        train_inst = 0.0
        for label in range(n_clusters):
            n = (kmeans.labels_ == label).sum()
            cluster_idx = np.argwhere(kmeans.labels_ == label).squeeze().tolist()

            ds = RegressionDataset("placeholder")
            # confusing, but this sets the name of the new dataset to self so that it prints in the results files
            ds.set_name(self.__str__())
            ds.x = self.x[cluster_idx]
            ds.y = self.y[cluster_idx]
            ds.prototype = torch.from_numpy(kmeans.cluster_centers_[label])

            if (
                train_inst / float(self.x.size(0)) > 0.75
                or (train_inst + n) / float(self.x.size(0)) == 1
            ):
                # add the rest of the instances to the test datasets.
                # TODO: handle the case where there is only one cluster
                test_idx += cluster_idx
                test_sets.append(ds)
            else:
                # add the cluster to the training set
                train_idx += cluster_idx
                train_inst += n
                train_sets.append(ds)

        # hack to get the test set of the clustered meta set contain all of the examples.
        # This is because it was hard to change the stats object in all of the models training files to take a variable length
        # based on many datasets
        test_set = test_sets[0]
        for s in test_sets[1:]:
            test_set.x = torch.cat((test_set.x, s.x))
            test_set.y = torch.cat((test_set.y, s.y))

        test_sets = [test_set]

        # clone self and set the x and y to that of the training indices so we can standard
        # normalize everythhing based off of the training indices.
        dataset = copy.deepcopy(self)
        dataset.x = dataset.x[train_idx]
        dataset.y = dataset.y[train_idx]

        # normalize the training and the test set based off of the normalization parameters of training set
        x_mu, x_sigma, y_mu, y_sigma = dataset.standard_normalize()
        for s in train_sets:
            s.standard_normalize(x_mu, x_sigma, y_mu, y_sigma)

        for s in test_sets:
            s.standard_normalize(x_mu, x_sigma, y_mu, y_sigma)

        assert len(train_sets) > 0, "meta train sets should be more than 0"
        assert len(test_sets) > 0, "met test tasks should be greater than 0"

        return (
            [
                DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,)
                for ds in train_sets
            ],
            [DataLoader(ds, batch_size=batch_size, num_workers=4) for ds in test_sets],
        )

    def combined_clustered_dataset(
        self, n_clusters: int, batch_size: int, seed: int, get_val: bool = False
    ) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
        """
        This calls the meta clusters function and then recombines them into one traditional training and test set split.
        It was made in this way so that we could evaluate the same dataset splits on different models.
        """
        # when we want to compare this method to a normal model with a train and test st we need
        # to regroup the meta sets together

        train_sets, test_sets = self.clustered_datasets_shift(
            n_clusters, batch_size, seed
        )

        train = train_sets[0].dataset
        for dl in train_sets[1:]:
            train.x = torch.cat((train.x, dl.dataset.x))  # type: ignore
            train.y = torch.cat((train.y, dl.dataset.y))  # type: ignore

        test = test_sets[0].dataset
        for dl in test_sets[1:]:
            test.x = torch.cat((test.x, dl.dataset.x))  # type: ignore
            test.y = torch.cat((test.y, dl.dataset.y))  # type: ignore

        val_set = None
        if get_val:
            perm = torch.randperm(train.x.size(0))  # type: ignore
            val_n = train.x.size(0) // 10  # type: ignore

            val = copy.deepcopy(train)
            val.x = val.x[perm[:val_n]]  # type: ignore
            val.y = val.y[perm[:val_n]]  # type: ignore

            train.x = train.x[perm[val_n:]]  # type: ignore
            train.y = train.y[perm[val_n:]]  # type: ignore

            val_set = DataLoader(val, batch_size=batch_size, shuffle=True)

        return (
            DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False),
            val_set,
            DataLoader(test, batch_size=batch_size),
        )

    def get_tsne_clusters(self, n_clusters: int, seed: int) -> Tuple[np.array, int]:
        """return cluster classes elementwise per instance, return the class which is the test set"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(self.x.numpy())
        test_cluster = self.get_test_cluster(n_clusters, 1.0 / n_clusters, kmeans)
        return kmeans.labels_, test_cluster

    def get_exclusive_sample(
        self, batch_size: int, excl_idx: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """this was made for sampling a series of points which are not in the query set"""
        # remove the unwanted indices
        perm = torch.randperm(len(self))
        # summing over a bool tensor performs like an or, so we will have [0, 1] tensor
        del_idx = (perm == excl_idx).sum(dim=0)
        perm = perm[del_idx != 1]

        return (
            self.x[perm[:batch_size]].to(device),
            self.y[perm[:batch_size]].to(device),
            perm[:batch_size],
        )

    def np_test_sample(
        self,
        batch_size: int,
        tgt_x: torch.Tensor,
        tgt_y: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, ...]:
        ctx_n = int(torch.randint(10, 20, (1,)).item())

        x_ctx = torch.zeros(batch_size, ctx_n, self.x.size(1)).to(device)
        y_ctx = torch.zeros(batch_size, ctx_n, 1).to(device)
        x_tgt = torch.zeros(batch_size, ctx_n + 1, self.x.size(1)).to(device)
        y_tgt = torch.zeros(batch_size, ctx_n + 1, 1).to(device)

        for b in range(batch_size):
            perm = torch.randperm(self.x.size(0))

            idx = perm[:ctx_n]
            x_sample, y_sample = self.x[idx], self.y[idx].unsqueeze(1)

            x_ctx[:, :] = x_sample
            y_ctx[:, :] = y_sample
            x_tgt[:, :ctx_n] = x_sample
            y_tgt[:, :ctx_n] = y_sample

        x_tgt[:, ctx_n, :] = tgt_x
        y_tgt[:, ctx_n, :] = tgt_y.unsqueeze(1)

        return x_ctx.to(device), y_ctx.to(device), x_tgt.to(device), y_tgt.to(device)

    def np_train_sample(
        self, batch_size: int, tgt_idx: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        """
        sample the query set and the target set from the same dataset, make sure that the query
        set is not part of the target set by passing in the tgt indices used
        """
        ctx_n = int(torch.randint(10, 20, (1,)).item())

        x_ctx = torch.zeros(batch_size, ctx_n, self.x.size(1)).to(device)
        y_ctx = torch.zeros(batch_size, ctx_n, 1).to(device)
        x_tgt = torch.zeros(batch_size, ctx_n + 1, self.x.size(1)).to(device)
        y_tgt = torch.zeros(batch_size, ctx_n + 1, 1).to(device)

        for b in range(batch_size):
            perm = torch.randperm(self.x.size(0))

            idx = perm[:ctx_n]
            # if the query point is in this permutation, get the next random set of permutations
            if (idx == tgt_idx[b]).sum() > 0:
                idx = perm[ctx_n : 2 * ctx_n]

            x_sample, y_sample = self.x[idx], self.y[idx].unsqueeze(1)

            x_ctx[:, :] = x_sample
            y_ctx[:, :] = y_sample
            x_tgt[:, :ctx_n] = x_sample
            y_tgt[:, :ctx_n] = y_sample

        x_tgt[:, ctx_n, :] = self.x[tgt_idx]
        y_tgt[:, ctx_n, :] = self.y[tgt_idx].unsqueeze(1)

        return x_ctx.to(device), y_ctx.to(device), x_tgt.to(device), y_tgt.to(device)


class PBPDataset(RegressionDataset):
    def __init__(
        self, *args: Any, x: torch.Tensor = None, y: torch.Tensor = None, name: str = ""
    ) -> None:
        """
        this is for the datasets from the MC Dropout repository which were first used
        in probabilistic backpropagation https://arxiv.org/abs/1502.05336
        """
        super(PBPDataset, self).__init__()

        if x is None or y is None or name == "":
            raise ValueError("kwargs needs to have x, y and name for PBP dataset")

        self.x = x
        self.y = y
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class NPPBPDataset(RegressionDataset):
    def __init__(
        self,
        *args: Any,
        x: torch.Tensor = None,
        y: torch.Tensor = None,
        ctx_rng: Tuple[int, int],
        name: str = "",
    ) -> None:
        """
        this is for the datasets from the MC Dropout repository which were first used
        in probabilistic backpropagation https://arxiv.org/abs/1502.05336
        this version is modified to better suit the NP style of getitem
        """
        super(NPPBPDataset, self).__init__()

        if x is None or y is None or name == "":
            raise ValueError(
                f"kwargs needs to have x, y and name for PBP dataset, x: {x}, y: {y}, name: {name}"
            )

        self.x = x
        self.y = y
        self.ctx_rng = ctx_rng
        self.name = name
        self.ctx_n = 0

    def set_ctx_n(self) -> None:
        self.ctx_n = int(
            torch.randint(self.ctx_rng[0], self.ctx_rng[1] + 1, (1,)).item()
        )

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        """
        sample the query set and the target set from the same dataset, make sure that the query
        set is not part of the target set by passing in the tgt indices used
        """

        x_ctx = torch.zeros(self.ctx_n, self.x.size(1))
        y_ctx = torch.zeros(self.ctx_n, 1)
        x_tgt = torch.zeros(self.ctx_n + 1, self.x.size(1))
        y_tgt = torch.zeros(self.ctx_n + 1, 1)

        perm = torch.randperm(self.x.size(0))

        idx = perm[: self.ctx_n]
        # if the query point is in this permutation, get the next random set of permutations
        if (idx == i).sum() > 0:
            idx = perm[self.ctx_n : 2 * self.ctx_n]

        x_sample, y_sample = self.x[idx], self.y[idx].unsqueeze(1)

        x_ctx = x_sample
        y_ctx = y_sample
        x_tgt[: self.ctx_n] = x_sample
        y_tgt[: self.ctx_n] = y_sample

        x_tgt[self.ctx_n] = self.x[i]
        y_tgt[self.ctx_n] = self.y[i]

        return x_ctx, y_ctx, x_tgt, y_tgt


class ClassificationDataset(Dataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(ClassificationDataset, self).__init__()
        self.x: torch.Tensor
        self.y: torch.Tensor

        # these are the parameters of the normalization
        self.mu: torch.Tensor
        self.sigma: torch.Tensor

        # the prototype of the data in this dataset
        self.prototype: torch.Tensor

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]

    def __len__(self) -> int:
        return self.x.size(0)

    def prune(self, idx: torch.Tensor) -> None:
        self.x = self.x[idx]
        self.y = self.y[idx]

    def set_name(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def get_feature_ranges(self) -> torch.Tensor:
        """
        get the feature ranges for all of the x features, this was originally used
        to determine the ranges of features for generating adversarial examples as done by
        deep ensembles paper https://arxiv.org/abs/1612.01474
        """
        return torch.abs(self.x.min(dim=0)[0] - self.x.max(dim=0)[0])

    def valid(self) -> None:
        if torch.any(torch.isinf(self.x)) or torch.any(torch.isnan(self.x)):
            raise ValueError("x has invalid values")
        elif torch.any(torch.isinf(self.y)) or torch.any(torch.isnan(self.y)):
            raise ValueError("y has invalid values")

    def one_normalize(
        self, max_x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """one normalize the dataset by dividing by the max value of x and y"""
        if max_x is None:
            self.max_x = torch.abs(self.x).max(dim=0)[0]

            # if maximum x value is zero, we want to avoid dividing by zero
            self.max_x[self.max_x == 0] = 1
            self.x /= max_x
            return (self.max_x,)

        self.max_x = max_x
        self.x /= max_x
        # returns a tuple just to keep the API the same between regression and classification datasets
        return (max_x,)  # type: ignore

    def standard_normalize(
        self, mu: Optional[torch.Tensor] = None, sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """standard normalize the dataset by ([x,y] - mu) / sigma"""
        if mu is None or sigma is None:
            self.mu = self.x.mean(dim=0)
            self.sigma = self.x.std(dim=0)
            self.sigma[self.sigma == 0] = 1

            if torch.any(self.sigma == 0):
                raise ValueError(
                    "sigma should not have zero values, see what is going on here"
                )
                self.sigma[self.sigma == 0] = 1

            self.x = (self.x - self.mu) / self.sigma
            if hasattr(self, "prototype"):
                self.prototype = (self.prototype - self.mu) / self.sigma

            self.valid()
            return self.mu, self.sigma

        self.mu = mu
        self.sigma = sigma

        self.x = (self.x - self.mu) / self.sigma
        if hasattr(self, "prototype"):
            self.prototype = (self.prototype - self.mu) / self.sigma

        self.valid()

        return mu, sigma

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        perm = torch.randperm(self.x.size(0))
        return self.x[perm[:n]], self.y[perm[:n]]

    def get_test_cluster(
        self, n_clusters: int, tgt_test_size: float, kmeans: KMeans
    ) -> int:
        closest_cluster, closest_count = 0, 0
        for i in range(n_clusters):
            ith_cluster_count = (kmeans.labels_ == i).sum()
            if abs(ith_cluster_count - tgt_test_size) < closest_count:
                closest_count = ith_cluster_count
                closest_cluster = i

        return closest_cluster

    def get_cluster_indices(self, n_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """get cluster indices, returns indices of the train set and the test set"""
        kmeans = KMeans(n_clusters=n_clusters).fit(self.x.numpy())

        closest_cluster = self.get_test_cluster(n_clusters, 1.0 / n_clusters, kmeans)

        train_clusters = kmeans.labels_ != closest_cluster
        test_clusters = kmeans.labels_ == closest_cluster

        train_idx = np.argwhere(train_clusters).reshape(-1)
        test_idx = np.argwhere(test_clusters).reshape(-1)

        return torch.from_numpy(train_idx), torch.from_numpy(test_idx)

    def meta_clusters(
        self, n_clusters: int, batch_size: int, seed: int
    ) -> Tuple[List[Loader], ...]:
        """create n clusters of data for 'meta' datasets and return them as datasets"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(self.x.numpy())

        # these will be the returned training and testing tasks
        train_sets: List[ClassificationDataset] = []
        test_sets: List[ClassificationDataset] = []

        # these are the training and testing indices which we will use for normalization
        train_idx: np.array = []
        test_idx: np.array = []

        # total ratio of instances which have been used for training
        train_inst = 0.0
        for i, label in enumerate(range(n_clusters)):
            n = (kmeans.labels_ == label).sum()
            cluster_idx = np.argwhere(kmeans.labels_ == label).squeeze().tolist()

            ds = ClassificationDataset("none")
            # confusing, but this sets the name of the new dataset to self so that it prints in the results files
            ds.set_name(self.__str__())
            ds.x = self.x[cluster_idx]
            ds.y = self.y[cluster_idx]
            ds.prototype = torch.from_numpy(kmeans.cluster_centers_[i])

            if (
                train_inst / float(self.x.size(0)) > 0.75
                or (train_inst + n) / float(self.x.size(0)) == 1
            ):
                # add the rest of the instances to the test datasets.
                test_idx += cluster_idx
                test_sets.append(ds)
            else:
                # add the cluster to the training set
                train_idx += cluster_idx
                train_inst += n
                train_sets.append(ds)

        # hack to get the test set of the clustered meta set contain all of the examples.
        # This is because it was hard to change the stats object in all of the models training files to take a variable length
        # based on many datasets
        test_set = test_sets[0]
        for s in test_sets[1:]:
            test_set.x = torch.cat((test_set.x, s.x))
            test_set.y = torch.cat((test_set.y, s.y))

        test_sets = [test_set]

        # set the self x and y to be that of the training set so that when we normalize, everything works
        self.x = self.x[train_idx]
        self.y = self.y[train_idx]

        # normalize the training and the test set based off of the normalization parameters of training set
        x_mu, x_sigma = self.standard_normalize()
        for s in train_sets:
            s.standard_normalize(x_mu, x_sigma)

        for s in test_sets:
            s.standard_normalize(x_mu, x_sigma)

        assert len(train_sets) > 0, "meta train sets should be more than 0"
        assert len(test_sets) > 0, "met test tasks should be greater than 0"

        return (
            [
                Loader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
                for ds in train_sets
            ],
            [Loader(ds, batch_size=batch_size) for ds in test_sets],
        )

    def get_shifted_dataset(
        self, n_clusters: int, batch_size: int, seed: int, get_val: bool = False
    ) -> Tuple[Loader, Optional[Loader], Loader]:
        """create a dataset from the original which is clustered and shifted to form an OOD dataset"""

        train_sets, test_sets = self.meta_clusters(n_clusters, batch_size, seed)

        train = train_sets[0].dataset
        for dl in train_sets[1:]:
            train.x = torch.cat((train.x, dl.dataset.x))  # type: ignore
            train.y = torch.cat((train.y, dl.dataset.y))  # type: ignore

        test = test_sets[0].dataset
        for dl in test_sets[1:]:
            test.x = torch.cat((test.x, dl.dataset.x))  # type: ignore
            test.y = torch.cat((test.y, dl.dataset.y))  # type: ignore

        val_set = None
        if get_val:
            perm = torch.randperm(train.x.size(0))  # type: ignore
            val_n = train.x.size(0) // 10  # type: ignore

            val = copy.deepcopy(train)
            val.x = val.x[perm[:val_n]]  # type: ignore
            val.y = val.y[perm[:val_n]]  # type: ignore

            train.x = train.x[perm[val_n:]]  # type: ignore
            train.y = train.y[perm[val_n:]]  # type: ignore

            val_set = Loader(val, batch_size=batch_size, shuffle=True)  # type: ignore

        return (
            Loader(
                train, batch_size=batch_size, shuffle=True, drop_last=False  # type: ignore
            ),
            val_set,
            Loader(test, batch_size=batch_size),  # type: ignore
        )

    def get_tsne_clusters(self, n_clusters: int, seed: int) -> Tuple[np.array, int]:
        """return cluster classes elementwise per instance, return the class which is the test set"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(self.x.numpy())
        test_cluster = self.get_test_cluster(n_clusters, 1.0 / n_clusters, kmeans)
        return kmeans.labels_, test_cluster


DSet = Union[RegressionDataset, ClassificationDataset, PBPDataset]
