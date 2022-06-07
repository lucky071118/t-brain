from os import path
from typing_extensions import Self

from keras import Sequential, Input
from keras.layers import Dense, Activation
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.metrics import BinaryAccuracy
from keras.models import load_model
import numpy as np

from ..config import get_setting

setting = get_setting()


class ValidModel:
    def __init__(self: Self) -> None:
        self.model: Sequential

    def setup(self: Self) -> None:
        self.model = Sequential()
        self.model.add(Input(shape=(setting.doc2vec_model_size,)))

        for index, value in enumerate(setting.valid_model_layer):
            if index % 2 == 0:
                self.model.add(Dense(value))
            else:
                self.model.add(Activation(value))

        self.model.add(Dense(1))
        self.model.add(Activation("sigmoid"))
        self.model.summary()

    def compile(self: Self) -> None:
        self.model.compile(
            loss=BinaryCrossentropy(from_logits=False),
            optimizer=Adam(),
            metrics=[BinaryAccuracy()],
        )

    def fit(self: Self, dataset_x: np.ndarray, dataset_y: np.ndarray) -> None:
        self.model.fit(
            dataset_x,
            dataset_y,
            batch_size=setting.valid_model_batch,
            epochs=setting.valid_model_epoch,
        )

    def evaluate(self: Self, dataset_x: np.ndarray, dataset_y: np.ndarray) -> None:
        test_loss, test_acc = self.model.evaluate(dataset_x, dataset_y)
        print(f"test_loss: {test_loss}")
        print(f"test_acc: {test_acc}")

    def save(self: Self) -> None:
        self.model.save(setting.valid_model_file)

    def load(self: Self) -> None:
        self.model = load_model(setting.valid_model_file)

    def predict(self: Self, data_x: np.ndarray) -> np.ndarray:
        return self.model.predict(data_x)
