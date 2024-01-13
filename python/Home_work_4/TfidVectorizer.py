from TfidfTransformer import TfidfTransformer
from CountVectorizer import CountVectorizer
from typing import List


# с наследованием от CountVectorizer
class TfidVectorizer(CountVectorizer):

    def __init__(self):
        super().__init__()
        self.transformer = TfidfTransformer()

    def fit_transform(self, corpus: List[str]):
        count_matrix = super().fit_transform(corpus)
        return self.transformer.fit_transform(count_matrix)


# без наследованя от CountVectorizer
class TfidVectorizer2:

    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()

    def fit_transform(self, corpus: List[str]):
        count_matrix = self.vectorizer.fit_transform(corpus)
        return self.transformer.fit_transform(count_matrix)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()
