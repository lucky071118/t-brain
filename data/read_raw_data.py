"""Get data from json file"""
from os import path
from json import load
from typing import Generator, Dict


def get_raw_dataset() -> Generator[Dict, None, None]:
    """return generator"""
    file_name = "sample.json"
    file_path = path.join(path.dirname(__file__), file_name)
    raw_data_set = _read_file(file_path)
    for raw_data in raw_data_set:
        yield raw_data


def _read_file(file_name: str) -> dict:
    with open(file_name, encoding="UTF8") as json_file:
        return load(json_file)
