import pytest
from pizza import Pizza, Margherita, Pepperoni, Hawaiian


@pytest.mark.parametrize(
    "pizza, expected",
    [
        (Pepperoni('M'),
         f'Name: {Pepperoni("M").name}\nRecipe: {Pepperoni("M").recipe}\nSize: {Pepperoni("M").size}'),
        (Margherita('M'),
         f'Name: {Margherita("M").name}\nRecipe: {Margherita("M").recipe}\nSize: {Margherita("M").size}'),
        (Hawaiian('M'),
         f'Name: {Hawaiian("M").name}\nRecipe: {Hawaiian("M").recipe}\nSize: {Hawaiian("M").size}'),
        (Pepperoni('L'),
         f'Name: {Pepperoni("L").name}\nRecipe: {Pepperoni("L").recipe}\nSize: {Pepperoni("L").size}'),
        (Margherita('L'),
         f'Name: {Margherita("L").name}\nRecipe: {Margherita("L").recipe}\nSize: {Margherita("L").size}'),
        (Hawaiian('L'),
         f'Name: {Hawaiian("L").name}\nRecipe: {Hawaiian("L").recipe}\nSize: {Hawaiian("L").size}'),
    ],
)
def test_str(pizza, expected):
    assert pizza.__str__() == expected


def test_size_pizza_wrong_size():
    """
        Размер пиццы бывает только двух видов: M, L
    """

    with pytest.raises(ValueError):
        Pizza('XL')


def test_size_pepperoni_wrong_size():
    """
        Размер пиццы Pepperoni бывает только двух видов: M, L
    """

    with pytest.raises(ValueError):
        Pepperoni('XXL')


def test_size_margherita_wrong_size():
    """
        Размер пиццы бывает только двух видов: M, L
    """

    with pytest.raises(ValueError):
        Margherita('LM')


def test_size_hawaiian_wrong_size():
    """
        Размер пиццы бывает только двух видов: M, L
    """

    with pytest.raises(ValueError):
        Hawaiian('XL')


@pytest.mark.parametrize(
    "predicate, expected",
    [
        (Margherita('M') == Margherita('M'), True),
        (Pepperoni('M') == Pepperoni('M'), True),
        (Hawaiian('M') == Hawaiian('M'), True),
        (Pizza('M') == Pizza('M'), True),
        (Margherita('M') != Margherita('L'), True),
        (Pepperoni('M') != Margherita('M'), True),
        (Pizza('M') != Hawaiian('M'), True),
        (Pizza('M') != Pizza('L'), True),
        (Hawaiian('M') != Margherita('M'), True),
        (Pepperoni('M') != Hawaiian('M'), True),
    ],
)
def test_eq(predicate, expected):
    """
        Пицца идентична только в том случае, если рецепт и размер совпадают
    """
    assert predicate == expected


@pytest.mark.parametrize(
    "pizza, expected",
    [
        (Pepperoni('M'),
         {'Pepperoni 🍕': 'tomato sauce, mozzarella, pepperoni'}),
        (Margherita('M'),
         {'Margherita 🧀': 'tomato sauce, mozzarella, tomatoes'}),
        (Hawaiian('M'),
         {'Hawaiian 🍍': 'tomato sauce, mozzarella, chicken, pineapples'}),
        (Pepperoni('L'),
         {'Pepperoni 🍕': 'tomato sauce, mozzarella, pepperoni'}),
        (Margherita('L'),
         {'Margherita 🧀': 'tomato sauce, mozzarella, tomatoes'}),
        (Hawaiian('L'),
         {'Hawaiian 🍍': 'tomato sauce, mozzarella, chicken, pineapples'}),
    ],
)
def test_dict(pizza, expected):
    """
        Метод dict выводит словарь с рецептом
    """
    assert pizza.dict() == expected
