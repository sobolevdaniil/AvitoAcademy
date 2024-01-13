from decorators import bake, log


def test_bake():
    """
        Функция просто возвращает свой аргумент
    """
    actual = bake('pepperoni')
    expected = 'pepperoni'
    assert actual == expected


def test_log(capsys):
    """

    """

    @log('Foo')
    def foo():
        return 'Фуу'
    print(foo())
    captured = capsys.readouterr()

    assert captured.out == "\nFoo\nФуу\n"

