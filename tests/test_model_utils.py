import sys
import os

sys.path.append(f"{os.getcwd()}")

import torch
from collections import OrderedDict
from src.model.utils import (
    get_last_layer,
    get_num_classes,
    find_key_by_value,
)


model = OrderedDict(
    [
        (
            "model_layer_1",
            torch.tensor(
                [[0.2, -0.2, -0.1, -0.03, 0.17], [0.0, -0.1, -0.2, -0.3, -0.1]]
            ),
        ),
        (
            "model_layer_2",
            torch.tensor(
                [
                    [0.2, -0.2, -0.1, -0.0, 0.1],
                    [0.0, -0.1, -0.2, -0.3, -0.1],
                    [0.0, -0.1, -0.2, -0.3, -0.1],
                ]
            ),
        ),
    ]
)


def test_find_key_by_value():
    dictionary = {"a": 1, "b": 2, "c": 3}
    assert find_key_by_value(dictionary, 2) == "b"
    assert find_key_by_value(dictionary, 4) is None


def test_get_last_layer():
    assert get_last_layer(model) == "model_layer_2"


def test_get_num_classes():
    assert get_num_classes(model, "model_layer_2") == 3
