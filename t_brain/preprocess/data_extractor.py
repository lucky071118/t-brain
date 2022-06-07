"""Convert the dataset(x,y) to """
from enum import Enum
from typing import Generator, Tuple, Union, List, Optional
from random import randint

from ..data import get_raw_dataset


class Mode(Enum):
    """The mode of extraction"""

    POSITIVE = 1
    NEGATIVE = 2
    BOTH_AND = 3
    BOTH_OR = 4


def extract_dataset(mode: Mode) -> Generator[str, None, None]:
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
        if mode == Mode.POSITIVE and positive_data is not None:
            yield positive_data
        elif mode == Mode.NEGATIVE and negative_data is not None:
            yield negative_data
        elif (
            mode == Mode.BOTH_AND
            and positive_data is not None
            and negative_data is not None
        ):
            yield positive_data
            yield negative_data
        elif mode == Mode.BOTH_OR:
            if positive_data is not None:
                yield positive_data
            if negative_data is not None:
                yield negative_data


def _remove_duplicate_data_x(data_x_list: List[str]) -> List[str]:
    return list(set(data_x_list))


def _choose_data(
    data_x_list: List[str],
) -> Optional[str]:
    if len(data_x_list):
        index = randint(0, len(data_x_list) - 1)
        return data_x_list[index]
    return None
