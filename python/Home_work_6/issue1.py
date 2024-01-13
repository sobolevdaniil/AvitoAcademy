import sys
from datetime import datetime


original_write = sys.stdout.write


def my_write(string_text):
    if string_text != '\n':
        original_write(f'[{datetime.now()}]: {string_text}')
    else:
        original_write('\n')


sys.stdout.write = my_write

print('1, 2, 3')
print('1, 2, 3')
print('1, 2, 3')

sys.stdout.write = original_write

print('1, 2, 3')
print('1, 2, 3')
print('1, 2, 3')
