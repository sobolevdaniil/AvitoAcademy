import csv


def read_csv(path_of_file: str) -> list:
    """Функция записывает в список каждую строку csv файла"""
    result = []
    with open(path_of_file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            result.append(row[0].split(';'))
    return result
# не смог найти нужного разделителя, если будет время, ещё посмотрю


def search_deps(read_csv_file: list) -> list:
    """Поиск различных департаментов"""
    departments = list(set([employee[1] for employee in read_csv_file][1:]))
    return departments
# добавил comprehension, стало выглядеть лучше


def search_dep_teams(read_csv_file: list) -> dict:
    """Функция записывает словарь департамент и команды, входящие в него"""
    departments = search_deps(read_csv_file)
    dep_n_teams = {dep: ', '.join(list(set([employee[2] for employee in read_csv_file if employee[1] == dep])))
                   for dep in departments}
    return dep_n_teams
# как мне кажется, тут читабельность ухудшилась, стоит ли так делать? ниже другой вариант


def search_dep_teams_previous(read_csv_file: list) -> dict:
    """Функция записывает словарь департамент и команды, входящие в него"""
    departments = search_deps(read_csv_file)
    dep_n_teams = {}
    for dep in departments:
        commands = list(set([employee[2] for employee in read_csv_file if employee[1] == dep]))
        dep_n_teams[dep] = ', '.join(commands)
    return dep_n_teams


def dictionary_output(dictionary: dict):
    """Функция выводит в понятном виде содержимое словаря"""
    print()
    for key, value in dictionary.items():
        print("{0}: {1}".format(key, value))
    print()


def take_deps_info(read_csv_file: list) -> list:
    """Функция записывает в список департамент и информацию о нем:
       количество сотрудников, мин-макс зарплат, среднее зарплат"""
    info_about_deps = []
    departments = search_deps(read_csv_file)

    for dep in departments:
        list_of_salary = [int(employee[-1]) for employee in read_csv_file if employee[1] == dep]

        max_of_dep = max(list_of_salary)
        min_of_dep = min(list_of_salary)
        count = len(list_of_salary)
        sum_of_dep = sum(list_of_salary)

        list_of_info = [dep,
                        count,
                        str(min_of_dep) + '-' + str(max_of_dep),
                        int(sum_of_dep / count)]

        info_about_deps.append(list_of_info)
    return info_about_deps
# тут переписал нахождения min max count sum через питоновские функции, добавил comprehension


def deps_info_output(info_about_deps: list):
    """Функция выводит ифнормацию о департаменте"""
    print()
    for dep in info_about_deps:
        print('Название депаратамента:', dep[0])
        print('Количество сотрудников:', dep[1])
        print('Вилка зарплат min-max:', dep[2])
        print('Средняя зарплата:', dep[3], end='\n\n')


def write_info_to_csv(info: list, name_of_new_file: str = 'new_file'):
    """Функция записывает информацию о департаментах в csv файл"""
    with open(name_of_new_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(info)


def menu():
    """Функция вызывает меню программы"""
    while True:
        print('1: Вывести иерархию команд',
              '2: Вывести информацию о департаментaх',
              '3: Записать информацию о департаментах в csv файл',
              '0: Завершить', sep='\n')

        choice = int(input())
        while choice not in [0, 1, 2, 3]:
            print('Неверный ввод')
            choice = int(input())

        file_link = '/Users/daniilsobolev/Desktop/Avito/AAA/python/Home_work/Home_work_2/Corp_Summary.csv'
        file = read_csv(file_link)

        if choice == 1:
            teams_and_deps = search_dep_teams(file)
            dictionary_output(teams_and_deps)
        elif choice == 2:
            info_deps = take_deps_info(file)
            deps_info_output(info_deps)
        elif choice == 3:
            info_deps = take_deps_info(file)
            print('Введите название нового файла:')
            name = input()
            write_info_to_csv(info_deps, name)
            print()
        elif choice == 0:
            print('Завершение программы...')
            break
# тут избавился от повторов и переименовал choise в choice :) немного поменял логику while
# когда у нас функция не возвращает переменную, а сама что-то печатает,
# что можно написать в качестве -> выхода type-hinting


if __name__ == "__main__":

    menu()
