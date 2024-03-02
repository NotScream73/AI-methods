import itertools
import numpy as np


def calculate_fitness(chromosome, yields, costs):
    total_yield = 0
    total_cost = 0
    for i in range(len(chromosome)):
        total_yield += yields[i][chromosome[i]]
        total_cost += costs[chromosome[i]]
    return total_yield / total_cost, total_yield, total_cost


def rank_selection(population, fitness_values):
    ranks = np.argsort(fitness_values)
    selected_indices = ranks[len(ranks) // 2:]
    return population[selected_indices]


def genetic_algorithm(n, k, yields, costs, population_size=100, generations=1000, mutation=0.1):
    population = np.random.randint(k, size=(population_size, n))
    fitness_values = []
    for generation in range(generations):

        fitness_values = np.array([calculate_fitness(chromosome, yields, costs)[0] for chromosome in population])

        selected_population = rank_selection(population, fitness_values)

        crossover_points = np.random.randint(n, size=population_size // 2)
        offspring = np.empty_like(selected_population)
        for i in range(0, population_size // 2, 2):
            crossover_point = crossover_points[i]
            offspring[i, :] = np.concatenate(
                (selected_population[i, :crossover_point], selected_population[i + 1, crossover_point:]))
            offspring[i + 1, :] = np.concatenate(
                (selected_population[i + 1, :crossover_point], selected_population[i, crossover_point:]))

        for i in range(population_size // 2):
            for j in range(N):
                if np.random.rand() < mutation:
                    offspring[i, j] = np.random.randint(0, k)

        population = np.concatenate((population, offspring))

        fitness_values = np.array([calculate_fitness(chromosome, yields, costs)[0] for chromosome in population])
        best_indices = np.argsort(fitness_values)[-population_size:]
        population = population[best_indices]

    best_chromosome = population[np.argmax(fitness_values)]
    return calculate_fitness(best_chromosome, yields, costs)[1:], best_chromosome


def brute_force(n, k, yields, costs):
    best_fitness = 0
    best_harvest = 0
    best_cost = 0
    best_chromosome = None
    for chromosome in itertools.product(range(k), repeat=n):
        fitness, harvest, cost = calculate_fitness(chromosome, yields, costs)

        if fitness >= best_fitness:
            best_fitness = fitness
            best_harvest = harvest
            best_cost = cost
            best_chromosome = chromosome

    return best_harvest, best_cost, best_chromosome


def generate_data():
    N = np.random.randint(2, 5)
    k = np.random.randint(2, 15)

    yields = np.random.randint(5, 100, size=(N, k))
    costs = np.random.randint(5, 100, size=k)

    return N, k, yields.tolist(), costs.tolist()


def print_result(str_algorithm, harvest, cost, chromosome):
    print()
    print(str_algorithm)
    print("Урожайность:", harvest)
    print("Закупочная стоимость:", cost)
    print("Лучший вариант:", chromosome)


def print_data(n, k, yields, costs):
    print("Сгенерированные данные:")
    print()
    print("Количество полей:", n)
    print("Количество культур:", k)
    for i in range(len(yields)):
        print()
        for j in range(len(yields[i])):
            print(f"Поле {i + 1}. Урожайность культуры {j + 1}: {yields[i][j]}")
        print()
    for i in range(len(costs)):
        print(f"Закупочная стоимость культуры {i + 1}: {costs[i]}")


N, k, yields, costs = generate_data()

print_data(N, k, yields, costs)

brute_force_answer = brute_force(N, k, yields, costs)
print_result("Полный перебор.", brute_force_answer[0], brute_force_answer[1], brute_force_answer[2])

best_solution = genetic_algorithm(N, k, yields, costs)
print_result("Генетический алгоритм.", best_solution[0][0], best_solution[0][1], best_solution[1])
