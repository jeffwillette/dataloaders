import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np  # type: ignore
import torch
import torchvision  # type: ignore
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

from dataloaders import uci
from dataloaders.datasets import DSet, Loader, PBPDataset  # type: ignore
from dataloaders.real_estate import get_re_datasets  # type: ignore


def normalize(datasets: List[DSet], std: bool = True) -> None:
    """take a list of the regression datasets and normalize them. This mutates the datasets in place"""
    if std:
        params = datasets[0].standard_normalize()
        for d in datasets[1:]:
            d.standard_normalize(*params)

        return

    params = datasets[0].one_normalize()
    for d in datasets[1:]:
        d.one_normalize(*params)


cifar_tx = transforms.Compose(
    [
        transforms.RandomChoice(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
            ]
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)

svhn_tx = transforms.Compose(
    [
        transforms.RandomChoice(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
            ]
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ]
)

UCIData = Union[
    uci.Autos,
    uci.Wine,
    uci.ConcreteStrength,
    uci.Energy,
    uci.Housing,
    uci.MPG,
    uci.Protein,
    uci.Yacht,
    uci.Abalone,
    uci.BreastCancer,
    uci.CarEvaluation,
    uci.CensusIncome,
    uci.AirQuality,
]

train_size = 0.9


def get_ssl_sets(
    DS: Type[UCIData],
    datapath: str,
    num_labels: int,
    batch_size: int = 32,
    train_ratio: float = 0.9,
    std_norm: bool = True,
    **kwargs: Any,
) -> Tuple[Loader, ...]:
    l_train = DS(datapath, **kwargs)
    u_train = DS(datapath, **kwargs)
    test = DS(datapath, **kwargs)

    # validation set should be .1 of the training set.
    perm = torch.randperm(l_train.x.size(0))
    train_n = int(u_train.x.size(0) * train_size)

    l_train.prune(perm[:num_labels])
    u_train.prune(perm[num_labels:train_n])
    test.prune(perm[train_n:])

    if std_norm:
        params = l_train.standard_normalize()
        _ = u_train.standard_normalize(*params)
        _ = test.standard_normalize(*params)
    else:
        params = l_train.one_normalize()
        _ = u_train.one_normalize(*params)
        _ = test.one_normalize(*params)

    return (
        Loader(l_train, batch_size=batch_size, shuffle=True, drop_last=True),
        Loader(u_train, batch_size=batch_size, shuffle=True, drop_last=True),
        Loader(test, batch_size=batch_size, shuffle=True, drop_last=True),
    )


def get_ts_sets(
    DS: Type[UCIData],
    datapath: str,
    batch_size: int,
    get_val: bool = False,
    **kwargs: Any,
) -> Tuple[Optional[Loader], ...]:
    if not get_val:
        train = DS(datapath, **kwargs)
        test = DS(datapath, **kwargs)

        idx = torch.linspace(0, len(train), len(train) + 1).long()
        tr_idx = int(len(train) * train_size)

        train.prune(idx[:tr_idx])
        test.prune(idx[tr_idx:])

        max_x, max_y = train.one_normalize()
        _, _ = test.one_normalize(max_x, max_y)

        return (
            Loader(train, batch_size=batch_size, shuffle=True, drop_last=True),
            None,
            Loader(test, batch_size=batch_size, shuffle=True, drop_last=True),
        )

    train = DS(datapath, **kwargs)
    val = DS(datapath, **kwargs)
    test = DS(datapath, **kwargs)

    idx = torch.linspace(0, len(train), len(train) + 1)
    t_n = int(train.x.size(0) * 0.70)
    v_n = int(train.x.size(0) * 0.1)  # .5 of .5 would be .25

    train.prune(idx[:t_n])
    val.prune(idx[t_n : t_n + v_n])
    test.prune(idx[t_n + v_n :])

    max_x, max_y = train.one_normalize()
    _, _ = val.one_normalize(max_x, max_y)
    _, _ = test.one_normalize(max_x, max_y)

    return (
        Loader(train, batch_size=batch_size, shuffle=True, drop_last=True),
        Loader(val, batch_size=batch_size, shuffle=True, drop_last=True),
        Loader(test, batch_size=batch_size, shuffle=True, drop_last=True),
    )


pbp_sets = [
    "boston-housing",
    "concrete",
    "energy",
    "kin8nm",
    "naval-propulsion-plant",
    "power-plant",
    "wine-quality-red",
    "yacht",
    "protein-tertiary-structure",
]


def get_raw_pbp(name: str, datadir: str) -> DSet:
    if name not in pbp_sets:
        raise ValueError(f"{name} is an unknown pbp dataset")

    # load data
    path = os.path.join(datadir, "UCI_Datasets", name, "data", "data.txt")
    data = torch.from_numpy(np.loadtxt(path)).float()

    x = data[:, :-1]
    y = data[:, -1]

    return PBPDataset(x=x, y=y, name=name)


def get_pbp_sets(
    name: str, datadir: str, batch_size: int, get_val: bool = True
) -> Tuple[Loader, Optional[Loader], Loader]:
    """
    retrieves the datasets which were used in the following papers
    http://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf
    https://arxiv.org/abs/1502.05336
    https://arxiv.org/abs/1506.02142
    """
    if name not in pbp_sets:
        raise ValueError(f"{name} is an unknown pbp dataset")

    # load data
    path = os.path.join(datadir, "UCI_Datasets", name, "data", "data.txt")
    data = torch.from_numpy(np.loadtxt(path)).float()

    # make a random split of train and test
    idx_perm = torch.randperm(data.size(0))
    train_idx = int(data.size(0) * 0.9)

    if not get_val:
        # extract the features and labels
        train_ft = data[idx_perm[:train_idx], :-1]
        train_label = data[idx_perm[:train_idx], -1]

        test_ft = data[idx_perm[train_idx:], :-1]
        test_label = data[idx_perm[train_idx:], -1]

        train = PBPDataset(x=train_ft, y=train_label, name=name)
        test = PBPDataset(x=test_ft, y=test_label, name=name)
        params = train.standard_normalize()
        test.standard_normalize(*params)

        return (
            Loader(train, shuffle=True, batch_size=batch_size),
            None,
            Loader(test, batch_size=batch_size),
        )

    val_n = train_idx // 10

    # extract the features and labels
    train_ft = data[idx_perm[: train_idx - val_n], :-1]
    val_ft = data[idx_perm[train_idx - val_n : train_idx], :-1]

    train_label = data[idx_perm[: train_idx - val_n], -1]
    val_label = data[idx_perm[train_idx - val_n : train_idx], -1]

    test_ft = data[idx_perm[train_idx:], :-1]
    test_label = data[idx_perm[train_idx:], -1]

    train = PBPDataset(x=train_ft, y=train_label, name=name)
    val = PBPDataset(x=val_ft, y=val_label, name=name)
    test = PBPDataset(x=test_ft, y=test_label, name=name)

    params = train.standard_normalize()
    val.standard_normalize(*params)
    test.standard_normalize(*params)

    return (
        Loader(train, shuffle=True, batch_size=batch_size),
        Loader(val, shuffle=True, batch_size=batch_size),
        Loader(test, batch_size=batch_size),
    )


def get_uci_shifted_regression_sets(
    DS: Type[UCIData],
    datapath: str,
    batch_size: int,
    drop_last: bool = False,
    get_val: bool = False,
    std_norm: bool = True,
    **kwargs: Any,
) -> Tuple[Optional[Loader], ...]:
    if not get_val:
        train = DS(datapath, **kwargs)
        test = DS(datapath, **kwargs)

        train_idx, test_idx = train.get_cluster_indices(4)

        train.prune(train_idx)
        test.prune(test_idx)

        if std_norm:
            params = train.standard_normalize()
            _ = test.standard_normalize(*params)
        else:
            max_x, max_y = train.one_normalize()
            _, _ = test.one_normalize(max_x, max_y)

        return (
            Loader(train, batch_size=batch_size, shuffle=True, drop_last=True),
            None,
            Loader(test, batch_size=batch_size, shuffle=True, drop_last=True),
        )

    train = DS(datapath, **kwargs)
    val = DS(datapath, **kwargs)
    test = DS(datapath, **kwargs)

    perm = torch.randperm(train.x.size(0))
    t_n = int(train.x.size(0) * 0.5)
    v_n = int((train.x.size(0) - t_n) * 0.5)  # .5 of .5 would be .25

    train.prune(perm[:t_n])
    val.prune(perm[t_n : t_n + v_n])
    test.prune(perm[t_n + v_n :])

    if std_norm:
        params = train.standard_normalize()
        _ = val.standard_normalize(*params)
        _ = test.standard_normalize(*params)
    else:
        max_x, max_y = train.one_normalize()
        _, _ = val.one_normalize(max_x, max_y)
        _, _ = test.one_normalize(max_x, max_y)

    return (
        Loader(train, batch_size=batch_size, shuffle=True, drop_last=drop_last),
        Loader(val, batch_size=batch_size, shuffle=True, drop_last=drop_last),
        Loader(test, batch_size=batch_size, shuffle=True, drop_last=drop_last),
    )


def get_uci_sets(
    DS: Type[DSet],
    Loader: Type[Loader],
    datapath: str,
    batch_size: int,
    drop_last: bool = False,
    get_val: bool = False,
    std_norm: bool = True,
    **kwargs: Any,
) -> Tuple[Optional[Loader], ...]:
    """get the uci datasets"""
    if not get_val:
        train = DS(datapath, **kwargs)
        test = DS(datapath, **kwargs)

        perm = torch.randperm(train.x.size(0))

        n = int(train.x.size(0) * train_size)

        train.prune(perm[:n])
        test.prune(perm[n:])

        normalize([train, test], std=std_norm)

        return (
            Loader(train, batch_size=batch_size, shuffle=True, drop_last=drop_last),
            None,
            Loader(test, batch_size=batch_size, shuffle=True, drop_last=drop_last),
        )

    train = DS(datapath, **kwargs)
    val = DS(datapath, **kwargs)
    test = DS(datapath, **kwargs)

    perm = torch.randperm(train.x.size(0))
    t_n = int(train.x.size(0) * 0.5)
    v_n = int((train.x.size(0) - t_n) * 0.5)  # .5 of .5 would be .25

    train.prune(perm[:t_n])
    val.prune(perm[t_n : t_n + v_n])
    test.prune(perm[t_n + v_n :])

    normalize([train, test], std=std_norm)
    return (
        Loader(train, batch_size=batch_size, shuffle=True, drop_last=drop_last),
        Loader(val, batch_size=batch_size, shuffle=True, drop_last=drop_last),
        Loader(test, batch_size=batch_size, shuffle=True, drop_last=drop_last),
    )


# value tuple is (Dataset, path, kwargs, num_labels (for SSL))

data_args: Dict[str, Tuple[Type[UCIData], str, Optional[Dict[str, Any]], int]] = {
    "white-wine": (
        uci.Wine,
        "UCI/wine/winequality-white.csv",
        dict(variant="white"),
        440,
    ),
    "wine-red": (uci.Wine, "UCI/wine/winequality-red.csv", dict(variant="red"), 144),
    "auto": (uci.Autos, "UCI/autos/imports-85.data", None, 18),
    "mpg": (uci.MPG, "UCI/mpg/auto-mpg.data", None, 36),
    "housing": (uci.Housing, "UCI/housing/housing.data", None, 46),
    "concrete-strength": (
        uci.ConcreteStrength,
        "UCI/concrete-strength/Concrete_Data.xls",
        None,
        93,
    ),
    "energy": (uci.Energy, "UCI/energy-efficiency/ENB2012_data.xlsx", None, 70),
    "protein": (uci.Protein, "UCI/protein-tertiary/CASP.csv", None, 4116),
    "yacht": (uci.Yacht, "UCI/yacht-hydrodynamics/yacht_hydrodynamics.data", None, 28),
    "abalone": (  # abalone can be regression or classification
        uci.Abalone,
        "UCI/abalone/abalone.data",
        None,
        376,
    ),
    # these are for classification
    "car-evaluation": (uci.CarEvaluation, "UCI/car-evaluation/car.data", None, 172),
    "breast-cancer": (
        uci.BreastCancer,
        "UCI/breast-cancer/breast-cancer.data",
        None,
        28,
    ),
    "census-income": (uci.CensusIncome, "UCI/census-income/adult.data", None, 3556),
}


def get_ssl_dataset(
    name: str, datadir: str, batch_size: int, get_val: bool = False
) -> Tuple[DataLoader, ...]:

    if name in data_args.keys():
        DS = data_args[name][0]
        path = data_args[name][1]
        kwargs = data_args[name][2]
        labels = data_args[name][3]

        if kwargs is None:
            kwargs = {}

        return get_ssl_sets(
            DS, os.path.join(datadir, path), labels, batch_size, **kwargs
        )
    else:
        raise ValueError(f"{name} is an unknown dataset")


def get_classification_dataset(
    name: str, batch_size: int, datadir: str, get_val: bool = False
) -> Tuple[Optional[DataLoader], ...]:
    if name == "cifar10":
        train = torchvision.datasets.CIFAR10(datadir, train=True, transform=cifar_tx)
        val = torchvision.datasets.CIFAR10(datadir, train=True, transform=cifar_tx)

        idx = torch.randperm(len(train))
        train_idx = idx[: int(0.9 * len(train))]
        val_idx = idx[int(0.9 * len(train)) :]

        train_sampler = SubsetRandomSampler(train_idx.tolist())
        val_sampler = SubsetRandomSampler(val_idx.tolist())

        test = torchvision.datasets.CIFAR10(datadir, train=False, transform=cifar_tx)

        return (
            DataLoader(
                train,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=16,
                drop_last=True,
            ),
            DataLoader(
                val,
                batch_size=batch_size,
                num_workers=16,
                drop_last=True,
                sampler=val_sampler,
            ),
            DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=True),
        )
    elif name == "svhn":
        train = torchvision.datasets.SVHN(
            datadir, split="train", transform=svhn_tx, download=True
        )
        val = torchvision.datasets.SVHN(
            datadir, split="train", transform=svhn_tx, download=True
        )

        idx = torch.randperm(len(train))
        train_idx = idx[: int(0.9 * len(train))]
        val_idx = idx[int(0.9 * len(train)) :]

        train_sampler = SubsetRandomSampler(train_idx.tolist())
        val_sampler = SubsetRandomSampler(val_idx.tolist())

        test = torchvision.datasets.SVHN(
            datadir, split="test", transform=svhn_tx, download=True
        )

        return (
            DataLoader(
                train,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=4,
                drop_last=True,
            ),
            DataLoader(
                val,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=4,
                drop_last=True,
            ),
            DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=True),
        )
    else:
        raise ValueError(f"{name} is an unknown dataset")


def get_timeseries_sets(
    name: str, datadir: str, batch_size: int, get_val: bool = False
) -> Tuple[Optional[DataLoader], ...]:
    if name == "air-quality":
        return get_ts_sets(
            uci.AirQuality,
            os.path.join(datadir, "UCI/air-quality/AirQualityUCI.csv"),
            batch_size,
            get_val=get_val,
            seq_len=168,
            y_len=24,
        )
    else:
        raise ValueError(f"{name} is an unknown dataset")


def get_dataset_by_name(name: str, datadir: str) -> UCIData:
    """gets the datasets and not the dataloaders"""
    if name in data_args.keys():
        DS = data_args[name][0]
        path = data_args[name][1]
        kwargs = data_args[name][2]

        if kwargs is None:
            kwargs = {}

        return DS(os.path.join(datadir, path), **kwargs)

    raise ValueError(f"{name} is an unknown dataset")


def get_dist_shifted_sets_by_name(
    name: str,
    datadir: str,
    batch_size: int,
    get_val: bool = False,
    std_norm: bool = True,
) -> Tuple[Optional[Loader], ...]:

    DS = data_args[name][0]
    path = data_args[name][1]
    kwargs = data_args[name][2]

    if kwargs is None:
        kwargs = {}

    if name in data_args.keys():
        return get_uci_shifted_regression_sets(
            DS,
            os.path.join(datadir, path),
            batch_size,
            get_val=get_val,
            std_norm=std_norm,
            **kwargs,
        )
    else:
        raise ValueError(f"{name} is an unknown dataset")


def get_uci_by_name(
    name: str,
    datadir: str,
    batch_size: int,
    get_val: bool = False,
    std_norm: bool = True,
) -> Tuple[Optional[Loader], ...]:
    """get a set of dataloaders by name"""
    if name == "real-estate":
        return get_re_datasets(datadir, batch_size)  # type: ignore
    elif name in data_args.keys():
        DS = data_args[name][0]
        path = data_args[name][1]
        kwargs = data_args[name][2]

        if kwargs is None:
            kwargs = {}

        return get_uci_sets(
            DS,
            Loader,
            os.path.join(datadir, path),
            batch_size,
            get_val=get_val,
            std_norm=std_norm,
            **kwargs,
        )
    else:
        raise ValueError(f"{name} is an unknown dataset")
