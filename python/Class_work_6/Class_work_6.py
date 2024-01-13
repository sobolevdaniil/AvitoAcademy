from functools import partial


class Color:

    END = '\033[0'
    START = '\033[1;38;2'
    MOD = 'm'
    MIN_LEVEL = 0
    MAX_LEVEL = 255

    def __init__(self, red_level, green_level, blue_level):
        self._validate_level(red_level)
        self._validate_level(green_level)
        self._validate_level(blue_level)
        self.rgb = (red_level, green_level, blue_level)

    def __repr__(self):
        return f'{self.START};{self.rgb[0]};{self.rgb[1]};{self.rgb[2]}{self.MOD}●{self.END}{self.MOD}'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.rgb == other.rgb
        return False

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(*map(lambda x, y: min(self.MAX_LEVEL, x + y), self.rgb, other.rgb))
        return False

    def __hash__(self):
        return hash(self.rgb)

    def __rmul__(self, contrast):
        return self.__class__(*map(partial(self._change_contrast, contrast), self.rgb))

    def _validate_level(self, level):
        if level < self.MIN_LEVEL or level > self.MAX_LEVEL:
            raise ValueError

    @staticmethod
    def _change_contrast(contrast, level):
        cl = - 256 * (1 - contrast)
        f = (259 * (cl + 255)) / (255 * (259 - cl))
        return int(f * (level - 128) + 128)


if __name__ == '__main__':
    a = Color(255, 250, 0)
    b = Color(0, 5, 0)
    c = Color(0, 100, 0)
    d = Color(0, 100, 0)
    print(c * a)
