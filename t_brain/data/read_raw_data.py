from json import load
from typing import Generator, Dict

from ..config import get_setting

setting = get_setting()


def get_raw_dataset() -> Generator[Dict, None, None]:
    """return generator"""
    raw_data_set = _read_file(str(setting.dataset_file))
    for raw_data in raw_data_set:
        yield raw_data


def _read_file(file_name: str) -> dict:
    with open(file_name, encoding="UTF8") as json_file:
        return load(json_file)
