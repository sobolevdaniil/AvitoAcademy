from morse import LETTER_TO_MORSE


def encode(message: str) -> str:
    """
    Кодирует строку в соответсвие с таблицей азбуки Морзе
    >>> encode('  ') # doctest: +NORMALIZE_WHITESPACE
    ' '
    >>> encode('SOS')
    '... ... ...'
    >>> encode(30)
    Traceback (most recent call last):
        ...
    TypeError: 'int' object is not iterable
    """
    encoded_signs = [
        LETTER_TO_MORSE[letter] for letter in message
    ]

    return ' '.join(encoded_signs)


# можно ли убрать предупреждение и корректно ли это?
# можно ли импортировать недостающее, а не дублировать?
