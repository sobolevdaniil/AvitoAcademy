from typing import Dict


class Pizza:
    """
        Класс Pizza - основной класс, с методами, которые будут у дочерних классов.
    """
    name = 'Pizza'
    possible_size = ['L', 'M']

    def __init__(self, size) -> None:
        if size not in self.possible_size:
            raise ValueError('Incorrect size')
        self.size = size
        self.recipe = ''

    def dict(self) -> Dict[str, str]:
        """
            Метод выводит рецепт пиццы в виде словаря
        """
        return {self.name: self.recipe}

    def __str__(self) -> str:
        """
             Метод подменяет метод print
        """
        return f'Name: {self.name}\nRecipe: {self.recipe}\nSize: {self.size}'

    def __eq__(self, other) -> bool:
        """
            Метод позволяет сравнивать пиццы через символ ==
        """
        if isinstance(other, self.__class__):
            return self.recipe == other.recipe and self.size == other.size
        return False


class Margherita(Pizza):
    """
        Класс для пиццы Маргарита
    """
    name = 'Margherita 🧀'

    def __init__(self, size: str) -> None:
        super().__init__(size)
        self.recipe = 'tomato sauce, mozzarella, tomatoes'


class Pepperoni(Pizza):
    """
        Класс для пиццы Пепперони
    """
    name = 'Pepperoni 🍕'

    def __init__(self, size: str) -> None:
        super().__init__(size)
        self.recipe = 'tomato sauce, mozzarella, pepperoni'


class Hawaiian(Pizza):
    """
        Класс для пиццы Гавайская
    """
    name = 'Hawaiian 🍍'

    def __init__(self, size: str) -> None:
        super().__init__(size)
        self.recipe = 'tomato sauce, mozzarella, chicken, pineapples'
