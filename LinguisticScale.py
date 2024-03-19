import matplotlib.pyplot as plt


def plot_scale(label_names, membership):
    plt.figure(figsize=(10, 6))
    for i, (parameters, belonging) in enumerate(membership):
        plt.plot(parameters, belonging, label=label_names[i])
    plt.title('Оценка убытков')
    plt.legend()
    plt.grid(True)
    plt.show()


def run():
    auto_generate = input("Использовать данные по умолчанию? (+/-)")
    if auto_generate[0] == '+':
        label_names = ["мало", "средне", "много"]
        membership = []
        membership.append(([0, 0, 2, 5], [1, 1, 1, 0]))
        membership.append(([3, 5, 8], [0, 1, 0]))
        membership.append(([6, 9, 10, 10], [0, 1, 1, 1]))
    elif auto_generate[0] == '-':
        num_labels = int(input("Введите количество оценок в шкале: "))
        label_names = input("Введите имена оценок в шкале (через запятую): ").split(',')
        membership = []
        for i in range(num_labels):
            function_type = input(
                f"Выберите функецию принадлежности для оценки {label_names[i]}. (треугольную/трапиецевидную)")
            if function_type.__contains__('тре'):
                parameters = input(
                    "Введите три параметра для треугольной функции принадлежности (через запятую): ").split(',')
                parameters = [int(param) for param in parameters]
                belonging = [0, 1, 0]
            elif function_type.__contains__('тра'):
                parameters = input(
                    "Введите четыре параметра для трапециевидной функции принадлежности (через запятую): ").split(',')
                parameters = [int(param) for param in parameters]
                belonging = [1 if parameters[0] == parameters[1] else 0, 1, 1,
                             1 if parameters[2] == parameters[3] else 0]
            else:
                print('Неизвестная функция принадлежности')
                return
            membership.append((parameters, belonging))
    else:
        print("Неизвестная операция")
        return
    plot_scale(label_names, membership)
