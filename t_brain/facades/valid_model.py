from typing import List
from sklearn.model_selection import train_test_split
import numpy as np

from ..preprocess import (
    extract_dataset,
    Mode,
    convert_bool_to_ndarray,
    convert_list_to_ndarray,
)
from ..model import ValidModel, Doc2VecModel


def train() -> None:
    valid_model = ValidModel()
    valid_model.setup()
    valid_model.compile()

    doc2vec_model = Doc2VecModel()
    doc2vec_model.load()

    dataset = extract_dataset(Mode.BOTH_AND)
    dataset_x = []
    dataset_y = []
    label = True
    for data in dataset:
        doc_vec = doc2vec_model.infer_vector(data)
        dataset_x.append(doc_vec)
        dataset_y.append(convert_bool_to_ndarray(label))
        label = not label

    dataset_x = convert_list_to_ndarray(dataset_x)
    dataset_y = convert_list_to_ndarray(dataset_y)

    (
        training_dataset_x,
        testing_dataset_x,
        training_dataset_y,
        testing_dataset_y,
    ) = train_test_split(dataset_x, dataset_y, test_size=0.2)

    valid_model.fit(training_dataset_x, training_dataset_y)

    valid_model.evaluate(testing_dataset_x, testing_dataset_y)

    valid_model.save()


def predict(documents: List[str]) -> np.ndarray:
    valid_model = ValidModel()
    valid_model.load()
    doc2vec_model = Doc2VecModel()
    doc2vec_model.load()
    document_vectors = []
    for document in documents:
        document_vectors.append(doc2vec_model.infer_vector(document))
    result = valid_model.predict(convert_list_to_ndarray(document_vectors))
    return result
