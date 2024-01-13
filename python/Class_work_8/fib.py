saved = dict()


def cash(func):
    def wrapper(n) -> int:
        if n in saved.keys():
            # print('saved')
            return saved[n]
        else:
            # print('new')
            saved[n] = func(n)
            return saved[n]
    return wrapper


@cash
def fi(n):
    if n <= 2:
        return 1
    return fi(n-1) + fi(n-2)


print(fi(3))
print(saved)