import pytest
from one_hot_encoder import fit_transform


def test_moscow():
    actual = fit_transform(['Moscow'])
    expected = [('Moscow', [1])]
    assert actual == expected


def test_moscow_new_york():
    actual = fit_transform(['Moscow', 'New York'])
    expected = [
        ('Moscow', [0, 1]),
        ('New York', [1, 0])
    ]
    assert actual == expected


def test_space():
    actual = fit_transform([''])
    expected = [('', [0])]
    assert actual != expected


def test_moscow_new_york_moscow():
    actual = fit_transform(['Moscow', 'New York', 'Moscow'])
    expected = [
        ('Moscow', [0, 0, 1]),
        ('New York', [0, 1, 0]),
        ('Moscow', [1, 0, 0])
    ]
    assert actual != expected


def test_moscow_interception_error():
    with pytest.raises(TypeError):
        fit_transform()
