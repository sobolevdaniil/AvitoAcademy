from collections import Counter
from typing import List
import math


class CountVectorizer:
    def __init__(self):
        self.vocabulary = None

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """Функция возвращает мешок слов в виде матрицы"""
        word_of_text_counter = []
        for text in corpus:
            word_of_text_counter.append(Counter(text.lower().split()))

        unique_words_in_corpus = []
        for counter_text in word_of_text_counter:
            for word in counter_text.keys():
                if word not in unique_words_in_corpus:
                    unique_words_in_corpus.append(word)
        self.vocabulary = unique_words_in_corpus

        matrix = [
            [counter_text.get(word, 0) for word in self.vocabulary]
            for counter_text in word_of_text_counter
        ]

        return matrix

    def get_feature_names(self) -> list:
        """Функция возвращает уникальные слова в корпусе"""
        if self.vocabulary is None:
            raise ValueError('No vocabulary')
        return self.vocabulary


class TfidfTransformer:

    @staticmethod
    def tf_transform(matrix: List[List[int]]) -> List[List[float]]:
        """Функция возвращает матрицу term frequency"""
        tf_matrix = []
        for row in matrix:
            new_row = []
            full_length = sum(row)
            for word in row:
                word /= full_length
                new_row.append(word)
            tf_matrix.append(new_row)

        return tf_matrix

    @staticmethod
    def idf_transform(matrix: List[List[int]]) -> List[float]:
        """Функция возвращает список inverse document-frequency"""
        cont_texts = len(matrix)
        idf_list = []
        for row in list(zip(*matrix)):
            a = math.log((cont_texts + 1) / (sum([int(num > 0) for num in row]) + 1)) + 1
            idf_list.append(a)

        return idf_list

    def fit_transform(self, matrix: List[List[int]]) -> List[List[float]]:
        tf = self.tf_transform(matrix)
        idf = self.idf_transform(matrix)

        tf_idf = [[x * y for x, y in zip(row, idf)] for row in tf]

        return tf_idf


class TfidVectorizer(CountVectorizer):
    def __init__(self):
        super().__init__()
        self.transformer = TfidfTransformer()

    def fit_transform(self, corpus: List[str]):
        count_matrix = super().fit_transform(corpus)  # super() получает объект родит класса
        return self.transformer.fit_transform(count_matrix)


class TfidVectorizer2:

    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()

    def fit_transform(self, corpus: List[str]):
        count_matrix = self.vectorizer.fit_transform(corpus)
        return self.transformer.fit_transform(count_matrix)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()


if __name__ == '__main__':

    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]

    vectorizer = TfidVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(count_matrix)
