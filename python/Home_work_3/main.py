from collections import Counter
from typing import List


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

    def get_feature_names(self) -> List[str]:
        """Функция возвращает уникальные слова в корпусе"""
        if self.vocabulary is None:
            raise ValueError('No vocabulary')
        return self.vocabulary


if __name__ == '__main__':

    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]

    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(count_matrix)
