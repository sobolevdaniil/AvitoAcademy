class Pokemon:
    def __init__(self, name, category, weaknesses: tuple):
        self.name = name
        self.category = category
        self._weaknesses = weaknesses

    @property
    def weaknesses(self):
        return self._weaknesses

    @weaknesses.setter
    def weaknesses(self, weaknesses):
        self._weaknesses = weaknesses

    def get_weaknesses(self):
        return self._weaknesses[:1]


if __name__ == '__main__':
    bulbasaur = Pokemon(name='Bulbasaur', category='seed', weaknesses=('fire', 'psychic', 'flying', 'ice'))
    bulbasaur.weaknesses = ('psychic', 'flying')
    print(bulbasaur.weaknesses)
