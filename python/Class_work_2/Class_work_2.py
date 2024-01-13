import random
import string
from typing import Sequence


def split_to_digits(number: int) -> list:
    """Функция разбивает исходное число на разряды"""
    if not isinstance(number, int) or number <= 0:
        raise TypeError
    list_of_digits = []
    while number > 0:
        list_of_digits.append(number % 10)
        number = number // 10
    return list(reversed(list_of_digits))


def subtract_lists(seq_1, seq_2: Sequence) -> list:
    """Функция находит разницу множеств"""
    set_of_seq_2 = set(seq_2)
    list_of_el = [el for el in seq_1 if el not in set_of_seq_2]
    return list_of_el


def validate_coords(*coords: tuple) -> list:
    """Вывести некорректно заданный точки"""
    list_of_errors = []
    for lat, long in coords:
        error = []
        if lat < -90 or lat > 90:
            error.append(f'Неверно задана широта: {lat} (должна быть от -90.0 до 90.0)')
        if long < -180 or long > 180:
            error.append(f'Неверно задана долгота: {long} (должна быть от -180.0 до 180.0)')
        if not error:
            continue
        list_of_errors.append({
            'lat': lat,
            'lng': long,
            'error': '; '.join(error)
        })
    return list_of_errors


def count_words(text: str) -> dict:
    """Функция считает количество различных слов в тексте"""
    result = {}
    for word in text.split(' '):
        if word not in result:
            word = word.lower()
            result[word] = 1
        else:
            result[word] += 1
    return result


def check_braces(text: str) -> int:
    """Функция находит неверно закрывающуюся скобку"""
    brackets = []
    brackets_map = {')': '(',
                    '}': '{',
                    ']': '['}
    for i, symbol in enumerate(text):
        if symbol in '{[(':
            brackets.append(symbol)
        elif symbol in '}])':
            if not brackets:
                return i
            last_bracket = brackets.pop()
            if brackets_map[symbol] != last_bracket:
                return i
    if brackets:
        return len(brackets)
    return -1


def generate_password(strength: int = 1, length: int = None, from_source: str = None) -> str:
    """Функция генерирует пароль"""
    symbols = ''
    if from_source:
        symbols = from_source
    else:
        if strength == 0:
            symbols = string.digits
        elif strength == 1:
            symbols = string.digits + string.ascii_lowercase
        elif strength == 2:
            symbols = string.digits + string.ascii_uppercase + string.ascii_lowercase
        elif strength == 3:
            symbols = string.digits + string.ascii_uppercase + string.ascii_lowercase + string.punctuation

    if not length:
        length = random.randint(4, 16)

    result = ''
    for i in range(length):
        result += random.choice(symbols)

    return result
