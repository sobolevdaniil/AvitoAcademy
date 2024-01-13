from time import sleep
from typing import Callable


def _percent(percent=0, width=30):
    left = width * percent // 100
    right = width - left
    print('\r[', "#" * left, ' ' * right, '] ', f'{percent:.0f}%', sep='', end='', flush=True)


def _progress(other_input):
    size = len(other_input)
    for i in other_input:
        _percent(100 * i // size)
        yield i
    _percent(100)
    print('')


def progress(func: Callable) -> Callable:
    def wrapper(some_input):
        func(_progress(some_input))
    return wrapper


@progress
def slow_func(some_input):
   print("Starting...")
   for _ in some_input:
       sleep(0.1)
   print("Done")


slow_func(range(100))
