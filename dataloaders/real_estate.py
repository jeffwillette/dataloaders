import os
from typing import Optional, Tuple

import numpy as np  # type: ignore
import torch

from dataloaders.datasets import Loader, RegressionDataset  # type: ignore


class REData(RegressionDataset):
    def __init__(self, datadir: str) -> None:
        x = np.load(os.path.join(datadir, "realestate", "x_13.npy"))
        y = np.load(os.path.join(datadir, "realestate", "y_13.npy"))
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]

    def __str__(self) -> str:
        return "NY Real Estate"


def get_re_datasets(
    datadir: str, batch_size: int, get_val: bool = False, std_norm: bool = True
) -> Tuple[Optional[Loader], ...]:
    if not get_val:
        train = REData(datadir)
        test = REData(datadir)

        perm = torch.randperm(train.x.size(0))

        n = int(train.x.size(0) * 0.9)

        train.prune(perm[:n])
        test.prune(perm[n:])

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

    train = REData(datadir)
    val = REData(datadir)
    test = REData(datadir)

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
        Loader(train, batch_size=batch_size, shuffle=True, drop_last=True),
        Loader(val, batch_size=batch_size, shuffle=True, drop_last=True),
        Loader(test, batch_size=batch_size, shuffle=True, drop_last=True),
    )
