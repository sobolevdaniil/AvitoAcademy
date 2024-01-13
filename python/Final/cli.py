import random
import click
from pizza import Pizza


pizza_types = [pizza.__name__.lower() for pizza in Pizza.__subclasses__()]


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option('--delivery', default=False, is_flag=True)
@click.argument('pizza', nargs=1)
def order(pizza: str, delivery: bool) -> None:
    """Prepares and delivers pizza"""

    print(f'\n__Order information__\n\n'
          f'Pizza: {pizza}')
    if pizza.lower() not in pizza_types:
        print('This pizza is not on the menu\n')
    else:
        print(f'Prepared for {random.randint(10, 15)}m')
        if delivery:
            print(f'Delivered for {random.randint(30, 60)}m!\n')
        else:
            print('Order cancelled!\n')


@cli.command()
def menu() -> None:
    """Shows menu"""

    print('\nMENU:\n')
    for pizza_ in pizza_types:
        print(
            f'Name: {pizza_}\n'
            f'Size: {"/".join(Pizza.possible_size)}\n'
            f'Cost: {random.randint(200, 300)}/{random.randint(300, 400)}\n'
        )
    print(f'Estimated delivery time: {random.randint(40, 75)}m\n')


if __name__ == '__main__':
    cli()
