from typing import List
from typing_extensions import Self

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
import jieba
import numpy as np

from ..config import get_setting

setting = get_setting()


class Doc2VecModel:
    def __init__(self: Self) -> None:
        self.documents: List = []
        self.model: Doc2Vec

    def save(self: Self) -> None:
        self.model.save(get_tmpfile(setting.doc2vec_model_file))

    def load(self: Self) -> None:
        self.model = Doc2Vec.load(get_tmpfile(setting.doc2vec_model_file))

    def setup_data(self: Self, sentence: str) -> None:
        word_list = []
        for word in jieba.cut(sentence, cut_all=False):
            word_list.append(word)
        self.documents.append(TaggedDocument(word_list, [len(self.documents)]))

    def train(self: Self) -> None:
        self.model = Doc2Vec(
            vector_size=setting.doc2vec_model_size,
            min_count=setting.doc2vec_model_min_count,
            epochs=setting.doc2vec_model_epoch,
        )
        self.model.build_vocab(self.documents)
        self.model.train(
            self.documents,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs,
        )

    def infer_vector(self: Self, sentence: str) -> np.ndarray:
        word_list = sentence.split(" ")
        return self.model.infer_vector(word_list)
