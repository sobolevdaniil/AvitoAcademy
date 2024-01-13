import sys
from functools import wraps


def redirect_output(filepath):
    def my_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with open(filepath, 'w') as f:
                sys.stdout = f
                result = func(*args, **kwargs)
            sys.stdout = sys.__stdout__
            return result

        return wrapper

    print('\n   Numbers in file "function_output.txt"')
    return my_decorator


@redirect_output('function_output.txt')
def calculate(a):
    for power in range(1, 5):
        for num in range(1, 20):
            print(num ** power, end=' ')
        print()
    return a


print(calculate(5))
print(calculate.__name__)
