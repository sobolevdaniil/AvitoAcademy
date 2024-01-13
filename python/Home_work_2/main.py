def read_csv(path_of_file: str) -> list:
    """Функция записывает в список каждую строку csv файла"""
    import csv
    result = []
    with open(path_of_file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            result.append(row[0].split(';'))
    return result


def search_deps(read_csv_file: list) -> list:
    """Поиск различных департаментов"""
    departments = []
    for employee in read_csv_file:
        if employee[1] not in departments:
            departments.append(employee[1])
    return departments[1:]


def search_dep_teams(read_csv_file: list) -> dict:
    """Функция записывает словарь департамент и команды, входящие в него"""
    departments = search_deps(read_csv_file)
    dep_n_teams = {}
    for dep in departments:
        spisok = []
        for employee in read_csv_file:
            if employee[1] == dep:
                if employee[2] not in spisok:
                    spisok.append(employee[2])
        dep_n_teams[dep] = ', '.join(spisok)
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
    departments = search_deps(read_csv_file)
    info_about_deps = []
    for dep in departments:
        list_of_info = [dep]
        count, sum_of_dep, max_of_dep, min_of_dep = 0, 0, 0, int(read_csv_file[1][-1])
        for employee in read_csv_file:
            if employee[1] == dep:
                count += 1
                if min_of_dep > int(employee[-1]):
                    min_of_dep = int(employee[-1])
                if max_of_dep < int(employee[-1]):
                    max_of_dep = int(employee[-1])
                sum_of_dep += int(employee[-1])
        list_of_info.append(count)
        list_of_info.append(str(min_of_dep) + '-' + str(max_of_dep))
        list_of_info.append(int(sum_of_dep / count))
        info_about_deps.append(list_of_info)
    return info_about_deps


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
    import csv
    with open(name_of_new_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(info)


def menu():
    choise = -1
    while choise != 0:
        print('1: Вывести иерархию команд',
              '2: Вывести информацию о департаментaх',
              '3: Записать информацию о департаментах в csv файл',
              '0: Завершить', sep='\n')

        choise = int(input())
        while choise not in [0, 1, 2, 3]:
            print('Неверный ввод')
            choise = int(input())

        if choise == 1:
            file_link = '/Users/daniilsobolev/Desktop/Avito/AAA/python/Home_work/Home_work_2/Corp_Summary.csv'
            file = read_csv(file_link)
            teams_and_deps = search_dep_teams(file)
            dictionary_output(teams_and_deps)
        elif choise == 2:
            file_link = '/Users/daniilsobolev/Desktop/Avito/AAA/python/Home_work/Home_work_2/Corp_Summary.csv'
            file = read_csv(file_link)
            info_deps = take_deps_info(file)
            deps_info_output(info_deps)
        elif choise == 3:
            file_link = '/Users/daniilsobolev/Desktop/Avito/AAA/python/Home_work/Home_work_2/Corp_Summary.csv'
            file = read_csv(file_link)
            info_deps = take_deps_info(file)
            print('Введите название нового файла:')
            name = input()
            write_info_to_csv(info_deps, name)
            print()

if __name__ == "__main__":

    menu()
