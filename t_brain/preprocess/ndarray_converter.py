from typing import List
import numpy as np


def convert_bool_to_ndarray(label: bool) -> np.ndarray:
    if label:
        return np.array([1], dtype="float32")
    return np.array([0], dtype="float32")


def convert_list_to_ndarray(input_: List) -> np.ndarray:
    return np.array(input_)
