import random


def triangular_membership_function(x, a, b, c):
    if a <= x <= b:
        return round((x - a) / (b - a), 2)
    elif b < x <= c:
        return round((c - x) / (c - b), 2)
    else:
        return 0.0


def fuzzy_union(set1, set2):
    union_set = []
    for val1, val2 in zip(set1, set2):
        union_set.append(max(val1, val2) if val1 is not None and val2 is not None else 0.0)
    return union_set


def run():
    num_sets = int(input("Введите количество нечетких множеств: "))
    sets_params = []
    for i in range(num_sets):
        print(f"\nМножество {i + 1}")
        a = float(input("Введите параметр 'a' для треугольной функции принадлежности: "))
        b = float(input("Введите параметр 'b' для треугольной функции принадлежности: "))
        c = float(input("Введите параметр 'c' для треугольной функции принадлежности: "))
        fuzzy_set = [a, b, c]
        operation = input("Случайная генерация: ")
        if operation == "+":
            count = int(input("Сколько чисел: "))
            crisp_values = [round(random.uniform(a - 2.0, c + 2.0), 2) for _ in range(count)]
        else:
            crisp_values = list(map(float, input("Введите четкие значения через пробел: ").split()))

        sets_params.append((fuzzy_set, crisp_values))

    membership_sets = []
    for fuzzy_set, crisp_values in sets_params:
        membership_set = []
        for val in crisp_values:
            membership_set.append(triangular_membership_function(val, *fuzzy_set))
        membership_sets.append(membership_set)

    union_result = membership_sets[0]
    for membership_set in membership_sets[1:]:
        union_result = fuzzy_union(union_result, membership_set)

    for i, (fuzzy_set, crisp_values) in enumerate(sets_params):
        print("\nМножество:", i + 1)
        print("Параметры треугольной функции принадлежности:", fuzzy_set)
        print("Четкие значения:", crisp_values)
        print("Функции принадлежности:", membership_sets[i])

    print("\nРезультат объединения:", union_result)
