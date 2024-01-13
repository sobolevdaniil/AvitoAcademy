from datetime import datetime
import sys


def timed_output(function):
    original_write = sys.stdout.write

    def my_write(string_text):
        if string_text != '\n':
            original_write(f'[{datetime.now()}]: {string_text}')
        else:
            original_write('\n')

    def wrapper(some_input):
        sys.stdout.write = my_write
        result = function(some_input)
        sys.stdout.write = original_write
        return result

    return wrapper


@timed_output
def print_greeting(name):
    print(f'Hello, {name}!')
    print(f'Hello, {name}!')
    return 5


print(print_greeting("Nikita"))
print_greeting("Nikita")
print_greeting("Nikita")
