class BaseAdvert:

    def __init__(self, dictionary):
        self.dictionary = {}
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = BaseAdvert(value)
            self.dictionary[key] = value

    def __getattr__(self, key):
        value = self.dictionary.get(key)
        return value


class ColorizeMixin:

    def __repr__(self):
        return f'\033[0;{self.repr_color_code};48m{self.title} | {self.price} ₽\033[0m'


class Advert(ColorizeMixin):
    repr_color_code = 32

    def __init__(self, dictionary):
        if 'price' not in dictionary:
            dictionary['price'] = 0
        elif dictionary['price'] < 0:
            raise ValueError
        if 'title' not in dictionary:
            raise ValueError
        self.vocabulary = BaseAdvert(dictionary)

    def __getattr__(self, key):
        return self.vocabulary.__getattr__(key)

    def __repr__(self):
        return super().__repr__()


lesson_str = {
    "class_": 0,
    "title": 'iphone',
    "location": {
        "address": "город Москва, Лесная, 7",
        "metro_stations": "Белорусская"
    }
}

ex = Advert(lesson_str)
print(ex)
