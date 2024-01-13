from typing import List
import math


class TfidfTransformer:

    @staticmethod
    def tf_transform(matrix: List[List[int]]) -> List[List[float]]:
        """Функция возвращает матрицу term frequency"""
        tf_matrix = []
        for row in matrix:
            full_length = sum(row)
            tf_matrix.append([word / full_length for word in row])

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
        """Функция возвращает матрицу значений tf-idf"""
        tf = self.tf_transform(matrix)
        idf = self.idf_transform(matrix)

        tf_idf = [[x * y for x, y in zip(row, idf)] for row in tf]

        return tf_idf
