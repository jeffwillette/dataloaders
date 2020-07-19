import os
from unittest import TestCase

import torch

from dataloaders.utils import get_uci_by_name  # type: ignore
from dataloaders.utils import data_args, get_pbp_sets, get_ssl_sets, pbp_sets

DATADIR = "/home/jeff/datasets"


class TestUCI(TestCase):
    @classmethod
    def setUpClass(self) -> None:
        """set datasets as a class attribute so we can test the class methods"""
        for name in data_args:
            print(f"loading datasets in setup: {name}")
            train, val, test = get_uci_by_name(name, DATADIR, 32)

            self.assertIsNotNone(train, "train cannot be none")  # type: ignore
            self.assertIsNotNone(test, "test cannot be none")  # type: ignore

            train, test, val = get_uci_by_name(name, DATADIR, 32, get_val=True)

            self.assertIsNotNone(train, "train cannot be none")  # type: ignore
            self.assertIsNotNone(val, "val cannot be none")  # type: ignore
            self.assertIsNotNone(test, "test cannot be none")  # type: ignore

            setattr(self, f"{name}_train", train)
            setattr(self, f"{name}_val", val)
            setattr(self, f"{name}_test", test)

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.types = ["train", "val", "test"]  # type: ignore

    def test_get_feature_ranges(self) -> None:
        for ds_type in self.types:
            for name in data_args:
                dl = getattr(self, f"{name}_{ds_type}")
                ranges = dl.dataset.get_feature_ranges()
                self.assertNotEqual(ranges.max().item(), 0)

    def test_pbp_sets(self) -> None:
        for name in pbp_sets:
            train, val, test = get_pbp_sets(name, DATADIR, 32)
            self.assertIsNotNone(train)
            self.assertIsNotNone(test)

    def test_get_ssl_set(self) -> None:
        for name in data_args:
            DS = data_args[name][0]
            path = data_args[name][1]
            kwargs = data_args[name][2]
            labels = data_args[name][3]

            if kwargs is None:
                kwargs = {}

            l_train, u_train, test = get_ssl_sets(
                DS, os.path.join(DATADIR, path), labels, 32, **kwargs
            )

            self.assertIsNotNone(l_train, "train cannot be none")
            self.assertIsNotNone(u_train, "train cannot be none")
            self.assertIsNotNone(test, "train cannot be none")

    def test_clustering_no_shift(self) -> None:
        for name in data_args:
            train = getattr(self, f"{name}_train")
            train_sets, test_sets = train.dataset.clustered_datasets_no_shift(3, 2, 0)

            self.assertIsNotNone(train_sets, "meta train set cannot be none")
            self.assertIsNotNone(test_sets, "meta test set cannot be none")

            # the training sets should be greater than one and the test sets should be exactly one
            self.assertGreater(len(train_sets), 1)
            self.assertEqual(len(test_sets), 1)

    def test_clustering_shift(self) -> None:
        for name in data_args:
            train = getattr(self, f"{name}_train")
            train_sets, test_sets = train.dataset.clustered_datasets_shift(3, 2, 0)

            self.assertIsNotNone(train_sets, "meta train set cannot be none")
            self.assertIsNotNone(test_sets, "meta test set cannot be none")

            # the training sets should be greater than one and the test sets should be exactly one
            self.assertGreater(len(train_sets), 1)
            self.assertEqual(len(test_sets), 1)

    # commented out because one normalize does the normalization in place which messes up the
    # standard normalization test. If this is to be used then it needs to be re-examined
    # def test_one_normalize(self) -> None:
    #     for name in data_args:
    #         train_set = getattr(self, f"{name}_train")
    #         params = train_set.dataset.one_normalize()

    #         self.assertTrue(torch.all(torch.abs(train_set.dataset.x) <= 1.0).item())

    #         for t in self.types:
    #             other_set = getattr(self, f"{name}_train")
    #             _ = other_set.dataset.one_normalize(*params)

    #             other_set.dataset.valid()

    def test_standard_normalize(self) -> None:
        for name in data_args:
            train_set = getattr(self, f"{name}_train")

            self.assertTrue(
                torch.all(torch.abs(train_set.dataset.x.mean(dim=0)) <= 0.01)
            )

            for t in self.types:
                other_set = getattr(self, f"{name}_train")
                self.assertTrue(
                    torch.all(torch.abs(other_set.dataset.x.mean(dim=0)) <= 0.01)
                )

                other_set.dataset.valid()

    def test_get_clusters(self) -> None:
        for name in data_args:
            first_total_len = 0
            for i in range(5):
                train = getattr(self, f"{name}_train")

                tr, val, test = train.dataset.combined_clustered_dataset(4, 10, i)

                ds_len = 0
                ds_len += len(tr.dataset)
                ds_len += len(test.dataset)

                if i == 0:
                    first_total_len = ds_len
                    continue

                self.assertEqual(first_total_len, ds_len)
