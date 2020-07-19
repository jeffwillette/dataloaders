from __future__ import annotations

import csv
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd  # type: ignore
import torch

from dataloaders.datasets import RegressionDataset  # type: ignore

symboling_lookup = {"-3": 0, "-2": 1, "-1": 2, "0": 3, "1": 4, "2": 5, "3": 6}
make_lookup = {
    "alfa-romero": 0,
    "audi": 1,
    "bmw": 2,
    "chevrolet": 3,
    "dodge": 4,
    "honda": 5,
    "isuzu": 6,
    "jaguar": 7,
    "mazda": 8,
    "mercedes-benz": 9,
    "mercury": 10,
    "mitsubishi": 11,
    "nissan": 12,
    "peugot": 13,
    "plymouth": 14,
    "porsche": 15,
    "renault": 16,
    "saab": 17,
    "subaru": 18,
    "toyota": 19,
    "volkswagen": 20,
    "volvo": 21,
}
fuel_type_lookup = {"diesel": 0, "gas": 1}
aspiration_lookup = {"std": 0, "turbo": 1}
num_of_doors_lookup = {"four": 0, "two": 1}
bodystyle_lookup = {
    "hardtop": 0,
    "wagon": 1,
    "sedan": 2,
    "hatchback": 3,
    "convertible": 4,
}

drive_wheels_lookup = {"4wd": 0, "fwd": 1, "rwd": 2}
engine_location_lookup = {"front": 0, "rear": 1}
engine_type = {
    "dohc": 0,
    "dohcv": 1,
    "l": 2,
    "ohc": 3,
    "ohcf": 4,
    "ohcv": 5,
    "rotor": 6,
}
num_of_cylinders_lookup = {
    "eight": 0,
    "five": 1,
    "four": 2,
    "six": 3,
    "three": 4,
    "twelve": 5,
    "two": 6,
}
fuel_system_lookup = {
    "1bbl": 0,
    "2bbl": 1,
    "4bbl": 2,
    "idi": 3,
    "mfi": 4,
    "mpfi": 5,
    "spdi": 6,
    "spfi": 7,
}

# 75 total features including the categorical ones
AUTO_TOT_FT = 81
ONE_HOT_FEATURES = [0, 2, 3, 4, 5, 6, 7, 8, 14, 15, 17]

# dictionary of the onehot feature number to the lookup table
# None values mean there is no lookup and the field is a continuous numerical value
GET_LOOKUP_TABLE = {
    0: symboling_lookup,
    1: None,
    2: make_lookup,
    3: fuel_type_lookup,
    4: aspiration_lookup,
    5: num_of_doors_lookup,
    6: bodystyle_lookup,
    7: drive_wheels_lookup,
    8: engine_location_lookup,
    9: None,
    10: None,
    11: None,
    12: None,
    13: None,
    14: engine_type,
    15: num_of_cylinders_lookup,
    16: None,
    17: fuel_system_lookup,
    18: None,
    19: None,
    20: None,
    21: None,
    22: None,
    23: None,
    24: None,
    25: None,
}


def make_one_hot(category: str, lookup: Dict[str, int]) -> torch.Tensor:
    """make a one hot vector for the make of the car"""
    out = torch.zeros(len(lookup))
    idx = lookup[category]
    out[idx] = 1
    return out


def make_binary(positive: str, actual: str) -> torch.Tensor:
    return torch.tensor([1.0 if positive == actual else 0.0])


def make_one_hot_list(category: str, lookup: List[str]) -> torch.Tensor:
    """
    make a one hot vector given the category and the list of all categories
    NOTE: there are two make one hot functions because I initially did it with a dict and then changed the method to use a list
    """
    out = torch.zeros(len(lookup))

    if category == "?":
        return out

    idx = lookup.index(category)
    out[idx] = 1
    return out


class Autos(RegressionDataset):
    def __init__(self, datapath: str, **kwargs: Any) -> None:
        self.datapath = datapath

        self.x = torch.Tensor()
        self.y = torch.Tensor()
        self.numerical_indices: List[int] = []

        with open(self.datapath, "r") as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                x = torch.Tensor()
                # until 01 because we want to use the last one as the y value
                for j, col in enumerate(row[:-1]):
                    lookup = GET_LOOKUP_TABLE[j]

                    if col == "?":
                        # if column is ? then we need to find if it is numeric or categorical and set it to 0
                        if lookup is None:
                            x = torch.cat((x, torch.tensor([0.0])))
                        else:
                            x = torch.cat((x, torch.zeros(len(lookup))))
                    else:
                        if lookup is None:
                            x = torch.cat((x, torch.tensor([float(col)])))
                        else:
                            x = torch.cat((x, make_one_hot(col, lookup)))

                self.x = torch.cat((self.x, x.unsqueeze(0)))
                yval = 0.0 if row[-1] == "?" else float(row[-1])
                self.y = torch.cat((self.y, torch.tensor([yval])))

        nz = self.y != 0
        self.y = self.y[nz]
        self.x = self.x[nz]

    def __str__(self) -> str:
        return "UCIAuto"

    def __repr__(self) -> str:
        return "UCIAuto"


class Wine(RegressionDataset):
    def __init__(self, datapath: str, variant: str = "red") -> None:
        self.datapath = datapath
        self.variant = variant

        self.x = torch.Tensor()
        self.y = torch.Tensor()

        with open(self.datapath, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=";")
            for i, row in enumerate(reader):
                if i == 0:
                    continue

                x = torch.zeros(11)
                # until 01 because we want to use the last one as the y value
                for j, col in enumerate(row[:-1]):
                    x[j] = float(col)

                # minus one because we skipped row zero (column names)
                self.x = torch.cat((self.x, x.unsqueeze(0)))
                self.y = torch.cat((self.y, torch.tensor([float(row[-1])])))

        nz = self.y != 0
        self.y = self.y[nz]
        self.x = self.x[nz]

    def __str__(self) -> str:
        return f"UCIWine: {self.variant}"

    def __repr__(self) -> str:
        return f"UCIWine: {self.variant}"


class ConcreteStrength(RegressionDataset):
    def __init__(self, datapath: str, **kwargs: Any) -> None:
        self.datapath = datapath

        arr = pd.read_excel(datapath).to_numpy()
        self.x = torch.from_numpy(arr[:, :8]).float()
        self.y = torch.from_numpy(arr[:, -1]).float()

        nz = self.y != 0
        self.y = self.y[nz]
        self.x = self.x[nz]

    def __str__(self) -> str:
        return "UCIConcreteStrength"

    def __repr__(self) -> str:
        return "UCIConcreteStrength"


# Concrete dropout https://github.com/yaringal/DropoutUncertaintyExps/blob/master/UCI_Datasets/energy/data/data.txt
# only uses the first y regression target, so I made this loader to only take that one into account


class Energy(RegressionDataset):
    def __init__(self, datapath: str, **kwargs: Any) -> None:
        self.datapath = datapath

        arr = pd.read_excel(datapath).to_numpy()
        self.x = torch.from_numpy(arr[:, :8]).float()
        self.y = torch.from_numpy(arr[:, 8]).float()

        nz = self.y != 0
        self.x = self.x[nz]
        self.y = self.y[nz]

    def __str__(self) -> str:
        return "UCIEnergy"

    def __repr__(self) -> str:
        return "UCIEnergy"


class Housing(RegressionDataset):
    def __init__(self, datapath: str, **kwargs: Any) -> None:
        self.datapath = datapath

        self.x = torch.Tensor()
        self.y = torch.Tensor()

        with open(self.datapath, "r") as f:
            line = f.readline()
            while line:
                out = torch.tensor([float(i) for i in line[:-1].split(" ") if i != ""])
                self.x = torch.cat((self.x, out[:-1].unsqueeze(0)))
                self.y = torch.cat((self.y, out[-1:]))

                line = f.readline()

        nz = self.y != 0
        self.x = self.x[nz]
        self.y = self.y[nz]

    def __str__(self) -> str:
        return "UCIHousing"

    def __repr__(self) -> str:
        return "UCIHousing"


class MPG(RegressionDataset):
    def __init__(self, datapath: str, **kwargs: Any) -> None:
        self.datapath = datapath

        self.x = torch.Tensor()
        self.y = torch.Tensor()
        with open(self.datapath, "r") as f:
            line = f.readline()
            i = 0
            while line:
                tmp = [i for i in line.split(" ") if i != ""]
                tmp = ["0.0" if i == "?" else i for i in tmp]
                tmp = tmp[:8]
                tmp[-1] = tmp[-1][0]
                t = torch.tensor([float(i) for i in tmp])

                self.x = torch.cat((self.x, t[1:].unsqueeze(0)))
                self.y = torch.cat((self.y, t[:1]))

                i += 1
                line = f.readline()

        nz = self.y != 0
        self.x = self.x[nz]
        self.y = self.y[nz]

    def __str__(self) -> str:
        return "UCIMPG"

    def __repr__(self) -> str:
        return "UCIMPG"


class Protein(RegressionDataset):
    def __init__(self, datapath: str, **kwargs: Any) -> None:
        self.datapath = datapath
        self.tot_ft = 9
        self.instances = 45730

        self.x = torch.zeros(self.instances, self.tot_ft)
        self.y = torch.zeros(self.instances)

        with open(self.datapath, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for i, row in enumerate(reader):
                if i == 0:
                    continue

                x = torch.zeros(self.tot_ft)
                # until 01 because we want to use the last one as the y value
                for j, col in enumerate(row[:-1]):
                    x[j] = float(col)

                # minus one because we skipped row zero (column names)
                self.x[i - 1] = x
                self.y[i - 1] = float(row[-1])

        nz = self.y != 0
        self.y = self.y[nz]
        self.x = self.x[nz]

    def __str__(self) -> str:
        return "UCIProtein"

    def __repr__(self) -> str:
        return "UCIProtein"


class Yacht(RegressionDataset):
    def __init__(self, datapath: str, **kwargs: Any) -> None:
        self.datapath = datapath

        self.x = torch.Tensor()
        self.y = torch.Tensor()

        with open(self.datapath, "r") as f:
            line = f.readline()
            i = 0
            while line:
                out = torch.tensor([float(i) for i in line[:-1].split(" ") if i != ""])
                self.x = torch.cat((self.x, out[:-1].unsqueeze(0)))
                self.y = torch.cat((self.y, out[-1:]))

                line = f.readline()
                i += 1

        nz = self.y != 0
        self.y = self.y[nz]
        self.x = self.x[nz]

    def __str__(self) -> str:
        return "UCIYacht"

    def __repr__(self) -> str:
        return "UCIYacht"


class Abalone(RegressionDataset):
    def __init__(self, datapath: str, **kwargs: Any) -> None:
        self.datapath = datapath

        self.x = torch.zeros(4177, 10)
        self.y = torch.zeros(4177)

        with open(self.datapath, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for i, row in enumerate(reader):
                x = torch.Tensor()
                # until -1 because we want to use the last one as the y value
                for j, col in enumerate(row[:-1]):
                    if j == 0:
                        x = torch.cat((x, make_one_hot(col, {"M": 0, "F": 1, "I": 2})))
                        continue

                    x = torch.cat((x, torch.tensor([float(col)])))

                # minus one because we skipped row zero (column names)
                self.x[i] = x
                self.y[i] = float(row[-1])

    def __str__(self) -> str:
        return "UCIAbalone"

    def __repr__(self) -> str:
        return "UCIAbalone"


class CarEvaluation(RegressionDataset):
    def __init__(self, datapath: str, **kwargs: Any) -> None:
        self.datapath = datapath

        self.x = torch.Tensor()
        self.y = torch.Tensor()

        self.cols = {
            0: {"vhigh": 0, "high": 1, "med": 2, "low": 3},
            1: {"vhigh": 0, "high": 1, "med": 2, "low": 3},
            2: {"2": 0, "3": 1, "4": 2, "5more": 3},
            3: {"2": 0, "4": 1, "more": 2},
            4: {"small": 0, "med": 1, "big": 2},
            5: {"low": 0, "med": 1, "high": 2},
            6: {"unacc": 0, "acc": 1, "good": 2, "vgood": 3},
        }

        with open(self.datapath, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for i, row in enumerate(reader):
                x = torch.Tensor()
                # until 01 because we want to use the last one as the y value
                for j, col in enumerate(row[:-1]):
                    x = torch.cat((x, make_one_hot(col, self.cols[j])))
                    continue

                # minus one because we skipped row zero (column names)
                self.x = torch.cat((self.x, x.unsqueeze(0)))
                self.y = torch.cat((self.y, make_one_hot(row[6], self.cols[6])))

    def __str__(self) -> str:
        return "UCICarEvaluation"

    def __repr__(self) -> str:
        return "UCICarEvaluation"


class BreastCancer(RegressionDataset):
    def __init__(self, datapath: str, **kwargs: Any) -> None:
        self.datapath = datapath

        self.x = torch.Tensor()
        self.y = torch.Tensor()

        self.binary = [6, 8, 10]
        # fmt: off
        self.cols = {
            2: {"10-19": 0, "20-29": 1, "30-39": 2, "40-49": 3, "50-59": 4, "60-69": 5, "70-79": 6, "80-89": 7, "90-99": 8},
            3: {"lt40": 0, "ge40": 1, "premeno" : 2},
            4: {"0-4": 0, "5-9": 1, "10-14": 2, "15-19": 3, "20-24": 4, "25-29": 5, "30-34": 6, "35-39": 7, "40-44": 8, "45-49": 9, "50-54": 10, "55-59": 11},
            5: {"0-2": 0, "3-5": 1, "6-8": 2, "9-11": 3, "12-14": 4, "15-17": 5, "18-20": 6, "21-23": 7, "24-26": 8, "27-29": 9, "30-32": 10, "33-35": 11, "36-39": 12},
            6: ["yes"],
            7: {"1": 0, "2": 1, "3": 2},
            8: ["left"],
            9: {"left_up": 0, "left_low": 1, "right_up": 2, "right_low": 3, "central": 4},
            10: ["yes"]
        }
        # fmt: on

        with open(self.datapath, "r") as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                x = torch.Tensor()
                # from 1 because the first column is the class
                for j, col in enumerate(row[1:]):
                    if col == "?":
                        # if column is ? then we need to find if it is numeric or categorical and set it to 0
                        # plus 2 because we skipped 1 index and the file starts counting from 1 ( the names file )
                        x = torch.cat((x, torch.zeros(len(self.cols[j + 2]))))
                    elif (j + 2) in self.binary:
                        x = torch.cat(
                            (x, make_binary(self.cols[j + 2][0], col))  # type: ignore
                        )
                    else:
                        x = torch.cat(
                            (x, make_one_hot(col, self.cols[j + 2]))  # type: ignore
                        )

                self.x = torch.cat((self.x, x.unsqueeze(0)))
                self.y = torch.cat((self.y, make_binary("recurrence-events", row[0])))

    def __str__(self) -> str:
        return "UCIBreastCancer"

    def __repr__(self) -> str:
        return "UCIBreastCancer"


class CensusIncome(RegressionDataset):
    def __init__(self, datapath: str, **kwargs: Any) -> None:
        self.datapath = datapath

        self.x = torch.zeros(32561, 104)
        self.y = torch.zeros(32561)

        self.continuous = [0, 2, 4, 10, 11, 12]
        self.categorical = [1, 3, 5, 6, 7, 8, 13]
        self.binary = [9]
        # fmt: off

        self.cols = {
            1: ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"],
            3: ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
            5: ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
            6: ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"],
            7: ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
            8: ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
            9: ["Male"],
            13: ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"],
            14: [">50K"]
        }
        # fmt: on

        with open(self.datapath, "r") as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if len(row) == 0:
                    continue
                x = torch.Tensor()
                # from 1 because the first column is the class
                for j, col in enumerate(row[:-1]):
                    col = col.strip()
                    if j in self.binary:
                        x = torch.cat(
                            (x, make_binary(self.cols[j][0], col))
                        )  # type: ignore
                    elif j in self.categorical:
                        x = torch.cat(
                            (x, make_one_hot_list(col, self.cols[j]))
                        )  # type: ignore
                    elif j in self.continuous:
                        c = torch.tensor([float(col) if col != "?" else 0.0])
                        x = torch.cat((x, c))

                self.x[i] = x
                self.y[i] = make_binary(self.cols[14][0], row[-1].strip())

    def __str__(self) -> str:
        return "UCICensusIncome"

    def __repr__(self) -> str:
        return "UCICensusIncome"


def positional_encoding(d: int, pos: torch.Tensor) -> torch.Tensor:
    """
    this should make the positional encoding specified in the Attention is All You Need paper
    https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model
    """

    if d % 2 != 0:
        raise ValueError(f"d should be divisible by 2: got {d}")

    # i refers to the ith pair of sine and cosine terms. d / 2 is the range of i. This makes
    # a tensor like [0, 0, 1, 1, 2, 2, ...]. After the final division all the elements should be
    # between 0, 1
    tmp = torch.zeros(d).to(pos.device)
    tmp[::2] = torch.linspace(0, (d / 2) - 1, int(d / 2))
    tmp[1::2] = torch.linspace(0, (d / 2) - 1, int(d / 2))
    tmp = tmp * 2 / d

    # repeat it into the full size. ignoring error because the size argument works as expected
    # getting the -1 dimension because if it is a batched tensor then we want the second to last dim
    tmp = tmp.repeat(pos.size()[-1], 1)  # type: ignore

    # set up the argument to the sine and cosine functions
    out = torch.zeros((*pos.size(), d)).to(pos.device)  # type: ignore
    out[:, :] = 10000
    out = out ** tmp
    pos = pos.unsqueeze(1)
    out = pos / out

    # call sine and cosine on alternating elements of each last dimension
    out[..., ::2] = torch.sin(out[..., ::2])
    out[..., 1::2] = torch.cos(out[..., 1::2])

    return out


class AirQuality(RegressionDataset):
    def __init__(self, datapath: str, **kwargs: Any) -> None:
        self.datapath = datapath

        self.x = torch.zeros(10000, 20)
        self.y = torch.zeros(10000)

        self.seq_len = kwargs["seq_len"]
        self.y_len = kwargs["y_len"]
        self.type = "timeseries"

        with open(self.datapath, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=";")
            i = 0
            for row in reader:
                if row[0] == "Date":
                    continue

                x = torch.zeros(20)
                y = float(row[5].strip().replace(",", ".") if row[5] != "" else 0)
                if y <= 0:
                    continue

                d = " ".join(row[:2])
                time = datetime.strptime(d, "%d/%m/%Y %H.%M.%S")
                x[0] = time.timestamp()

                curr = 8  # because the j's wont correspond to indices directly
                row = row[:-2]
                for j, col in enumerate(row):

                    if j in [0, 1, 5]:
                        continue

                    feature = (
                        float(col.strip().replace(",", ".")) if col.strip() != "" else 0
                    )

                    if feature == -200:
                        feature = 0

                    x[curr] = feature
                    curr += 1

                self.x[i] = x
                self.y[i] = y

                i += 1

        self.x = self.x[:i]
        self.y = self.y[:i]

        self.x[:, :8] = positional_encoding(8, self.x[:, 0])
        self.y = torch.log(self.y)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end = i + self.seq_len
        return self.x[i:end], self.y[end : end + self.y_len]

    def __len__(self) -> int:
        return int(self.x.size(0) - self.seq_len - self.y_len)

    def __str__(self) -> str:
        return "UCIAirQUality"

    def __repr__(self) -> str:
        return "UCIAirQuality"
