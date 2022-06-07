from ..preprocess import extract_dataset, Mode
from ..model import Doc2VecModel


def train() -> None:
    model = Doc2VecModel()
    dataset = extract_dataset(Mode.BOTH_OR)
    for data in dataset:
        model.setup_data(data)
    model.train()
    model.save()
