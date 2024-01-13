class BasePokemon1:
    def __init__(self, name, category):
        self.name = name
        self.category = category

    def to_str(self):
        return f'{self.name}/{self.category}'


class Pokemon1(BasePokemon1):
    def __init__(self, name, category, weaknesses):
        super().__init__(name, category)
        self.weaknesses = weaknesses


class BasePokemon:
    def __init__(self, name: str, category: str):
       self.name = name
       self.category = category

    def __str__(self):
       return f'{self.name}/{self.category}'


class EmojiMixin:
    ICON = {
        'grass': '1',
        'fire': '2',
        'water': '3',
        'electric': '4'
    }

    def __str__(self):
        text: str = super().__str__()
        for category, emoji in self.ICON.items():
            replaced = text.replace(category, emoji)
            if replaced != text:
                return replaced
        return text


class Pokemon(EmojiMixin, BasePokemon):
    pass


if __name__ == '__main__':
    pokemon = Pokemon(name='Bulbasaur', category='grass')
    print(pokemon)
