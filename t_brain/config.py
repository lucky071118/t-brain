from os import path
import sys
from pathlib import Path, WindowsPath, PosixPath
from ipaddress import IPv4Address
from functools import lru_cache
from typing import Dict, Any, List


from yaml import safe_load
from pydantic import BaseSettings, validator
from keras.layers import Activation, Dense


def _flatten_dict(dict_: Dict, parent_key="") -> Dict[str, Any]:
    items = []
    for key, value in dict_.items():
        new_key = parent_key + "_" + key if parent_key else key
        if isinstance(value, dict):
            items.extend(_flatten_dict(value, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


def _convert_file_path_by_os(file_path: str) -> Path:
    root = Path(__file__).parent.parent.resolve()
    if sys.platform == "win32":
        return WindowsPath(root).joinpath(file_path)
    return PosixPath(root).joinpath(file_path)


def _setting_yaml_source(_: BaseSettings) -> Dict[str, Any]:
    setting_yaml_path = path.join(path.dirname(path.dirname(__file__)), "setting.yml")
    with open(setting_yaml_path, "r", encoding="utf8") as stream:
        return _flatten_dict(safe_load(stream))


class Settings(BaseSettings):
    dataset_file: Path

    server_ip: IPv4Address
    server_port: int
    server_debug: bool

    doc2vec_model_size: int
    doc2vec_model_min_count: int
    doc2vec_model_epoch: int
    doc2vec_model_file: Path

    valid_model_batch: int
    valid_model_epoch: int
    valid_model_split: float
    valid_model_file: Path
    valid_model_layer: List

    @validator("dataset_file")
    @classmethod
    def dataset_file_is_path(cls, value: str):
        return _convert_file_path_by_os(value)

    @validator("doc2vec_model_size")
    @classmethod
    def doc2vec_size_greater_than_zero(cls, value: int):
        if value <= 0:
            raise ValueError("The size of doc2vec model must be greater than zero.")
        return value

    @validator("doc2vec_model_min_count")
    @classmethod
    def doc2vec_count_greater_than_two(cls, value: int):
        if value < 2:
            raise ValueError(
                "The minimum count of doc2vec model must be greater than two."
            )
        return value

    @validator("doc2vec_model_epoch")
    @classmethod
    def doc2vec_epoch_greater_than_zero(cls, value: int):
        if value <= 0:
            raise ValueError("The epoch of doc2vec model must be greater than zero.")
        return value

    @validator("doc2vec_model_file")
    @classmethod
    def doc2vec_file_is_path(cls, value: str):
        return _convert_file_path_by_os(value)

    @validator("valid_model_batch")
    @classmethod
    def valid_batch_greater_than_zero(cls, value: int):
        if value <= 0:
            raise ValueError("The batch of valid model must be greater than zero.")
        return value

    @validator("valid_model_epoch")
    @classmethod
    def valid_epoch_greater_than_zero(cls, value: int):
        if value <= 0:
            raise ValueError("The epoch of valid model must be greater than zero.")
        return value

    @validator("valid_model_split")
    @classmethod
    def valid_split_between_zero_and_one(cls, value: float):
        if 0 < value < 1:
            return value
        raise ValueError(
            "The split of training/testing data on valid model  must be between zero and one."
        )

    @validator("valid_model_file")
    @classmethod
    def valid_file_is_path(cls, value: str):
        return _convert_file_path_by_os(value)

    @validator("valid_model_layer")
    @classmethod
    def valid_layer_follow_keras_sequential_rule(cls, values: List):
        if len(values) % 2 == 1:
            raise ValueError(
                "Each dense layer must be combined with an activation layer."
            )

        for index, value in enumerate(values):
            if index % 2 == 0:
                assert Dense(
                    value
                ), "The layer of valid model must follow the sequential_rule of Keras"
            else:
                assert Activation(
                    value
                ), "The layer of valid model must follow the sequential_rule of Keras"

        return values

    class Config:
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                _setting_yaml_source,
                env_settings,
                file_secret_settings,
            )


@lru_cache
def get_setting() -> Settings:
    setting = Settings()
    return setting
