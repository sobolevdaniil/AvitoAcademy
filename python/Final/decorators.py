import random
from functools import wraps
from pizza import Margherita


def log(message: str):
    def decorator(func):
        """Decorator displaying function name and random number"""

        @wraps(func)
        def wrapper(*args):
            # print(f'\n{func.__name__} - {randint(10, 15)}m!')
            print()
            print(message.format(random.randint(15, 25)))
            result = func(*args)
            return result

        return wrapper

    return decorator


@log('Baked total for {}m')
def bake(pizza):
    """Cooking pizza"""
    return f'{pizza}'


bake(Margherita('M'))
