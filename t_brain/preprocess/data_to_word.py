"""Convert the dataset(x,y) to """
from typing import Generator, Tuple
from random import randint

from ..data import get_raw_dataset


def extract_dataset() -> Generator[Tuple, None, None]:
    """check whether x mapping to y"""
    raw_dataset = get_raw_dataset()
    for raw_data in raw_dataset:

        data_y = raw_data["ground_truth_sentence"]
        data_x_list = [
            sentence.replace(" ", "") for sentence in raw_data["sentence_list"]
        ]
        positive_data_x_list = _remove_duplicate_data_x(
            [data_x for data_x in data_x_list if data_x == data_y]
        )
        negative_data_x_list = _remove_duplicate_data_x(
            [data_x for data_x in data_x_list if data_x != data_y]
        )

        positive_data = _choose_data(positive_data_x_list)
        negative_data = _choose_data(negative_data_x_list)

        if positive_data is not None and negative_data is not None:
            yield (positive_data, negative_data)


def _remove_duplicate_data_x(data_x_list):
    return list(set(data_x_list))


def _choose_data(
    data_x_list,
):
    if len(data_x_list):
        index = randint(0, len(data_x_list) - 1)
        return data_x_list[index]
    return None
