import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time
import math
from sklearn.cluster import KMeans

# Paramètres de contrôle
temp_init = 10000
cooling = 0.995
temp_min = 0.0001
nb_trucks = 3
reheat_threshold = 0.001
reheat_value = 300
max_reheat_count = 6

# Chemin du fichier contenant la matrice de coûts
cost_matrix_file_path = 'vendor/coords/matrix.txt'


def read_coordinates():
    coordinates = []
    with open('vendor/Coords/list.txt', 'r') as file:
        for line in file:
            parts = line.strip().split()
            x = float(parts[1])
            y = float(parts[2])
            coordinates.append((x, y))
    return coordinates


def read_cost_matrix(file_path):
    cost_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [float(value) if value != 'nan' else np.nan for value in line.strip().split()]
            cost_matrix.append(row)
    return np.array(cost_matrix)


def generate_random_tour(cost_matrix):
    city_count = len(cost_matrix)
    return random.sample(range(city_count), city_count)


def distance(tour, cities):
    distance = 0
    for i in range(1, len(tour)):
        distance += np.linalg.norm(np.array(cities[tour[i - 1]]) - np.array(cities[tour[i]]))
    distance += np.linalg.norm(np.array(cities[tour[-1]]) - np.array(cities[0]))
    return distance


def split_tour(tour, num_trucks, cities):
    city_count = len(tour)
    city_coordinates = [cities[i] for i in tour]

    kmeans = KMeans(n_clusters=num_trucks, n_init=10, random_state=0).fit(city_coordinates)
    labels = kmeans.labels_

    sub_tours = [[] for _ in range(num_trucks)]
    for i, label in enumerate(labels):
        sub_tours[label].append(tour[i])

    for i in range(num_trucks):
        if sub_tours[i][0] != 0:
            sub_tours[i].insert(0, 0)
        if sub_tours[i][-1] != 0:
            sub_tours[i].append(0)

    return sub_tours


def simulated_annealing(cost_matrix, temp_init, cooling, temp_min, reheat_threshold, reheat_value, max_reheat_count):
    start_time = time.time()
    cities = read_coordinates()
    current_tour = generate_random_tour(cost_matrix)
    current_distance = distance(current_tour, cities)
    best_tour = current_tour
    best_distance = current_distance
    temperature = temp_init
    nb_iterations = 0
    reheat_count = 0

    while temperature > temp_min:
        nb_iterations += 1

        new_tour = copy.copy(current_tour)
        swap_indices = random.sample(range(1, len(new_tour)), 2)
        new_tour[swap_indices[0]], new_tour[swap_indices[1]] = new_tour[swap_indices[1]], new_tour[swap_indices[0]]

        new_distance = distance(new_tour, cities)
        delta_distance = new_distance - current_distance

        if delta_distance < 0 or random.random() < math.exp(-delta_distance / temperature):
            current_tour = new_tour
            current_distance = new_distance

        if current_distance < best_distance:
            best_tour = current_tour
            best_distance = current_distance

        temperature *= cooling

        if temperature < reheat_threshold and reheat_count < max_reheat_count:
            temperature += reheat_value
            reheat_count += 1

    sub_tours = split_tour(best_tour, nb_trucks, cities)

    total_distance = 0
    for i, sub_tour in enumerate(sub_tours):
        sub_tour_distance = distance(sub_tour, cities)
        print("Tour {}: Length {}, Route: {}".format(i + 1, sub_tour_distance, sub_tour))
        total_distance += sub_tour_distance

    print("Total distance: {}".format(total_distance))
    print("Execution time in seconds: ", time.time() - start_time)
    print("Number of iterations: {}".format(nb_iterations))

    # Modification ici: écrire les sous-tours dans le fichier composite_road.txt
    with open('vendor/coords_rec/composite_road.txt', 'w') as file:
        for i, sub_tour in enumerate(sub_tours):
            sub_tour_distance = distance(sub_tour, cities)
            print("Tour {}: Length {}, Route: {}".format(i + 1, sub_tour_distance, sub_tour))
            # Écriture du sous-tour
            for city_index in sub_tour:
                file.write(str(city_index) + '\n')
            # Écriture de la démarcation (série de tirets) entre les sous-tours
            if i < len(sub_tours) - 1:
                file.write('-----\n')
            total_distance += sub_tour_distance

    with open('vendor/coords_rec/road.txt', 'w') as file:
        for city_index in best_tour:
            file.write(str(city_index) + '\n')
    return best_tour, best_distance


def calculate_results(best_tour):
    cities_file_path = 'vendor/Coords_rec/road.txt'
    distances_file_path = 'vendor/Coords/distances.txt'
    times_file_path = 'vendor/Coords/times.txt'
    gas_file_path = 'vendor/Coords/gas.txt'
    gas_cost_file_path = 'vendor/Coords/gas_cost.txt'
    human_cost_file_path = 'vendor/Coords/human_cost.txt'

    with open(cities_file_path, 'r') as cities_file:
        cities = [int(line.strip()) for line in cities_file]

    with open(distances_file_path, 'r') as distances_file:
        distance_lines = [line.strip().split() for line in distances_file]
        distance_matrix = [[float(distance) for distance in line] for line in distance_lines]

    with open(times_file_path, 'r') as times_file:
        time_lines = [line.strip().split() for line in times_file]
        time_matrix = [[float(time) for time in line] for line in time_lines]

    with open(gas_file_path, 'r') as gas_file:
        gas_lines = [line.strip().split() for line in gas_file]
        gas_matrix = [[float(gas) for gas in line] for line in gas_lines]

    with open(gas_cost_file_path, 'r') as gas_cost_file:
        gas_cost_lines = [line.strip().split() for line in gas_cost_file]
        gas_cost_matrix = [[float(gas_cost) for gas_cost in line] for line in gas_cost_lines]

    with open(human_cost_file_path, 'r') as human_cost_file:
        human_cost_lines = [line.strip().split() for line in human_cost_file]
        human_cost_matrix = [[float(human) for human in line] for line in human_cost_lines]

    # Lire la matrice des distances
    with open('vendor/coords/distances.txt', 'r') as distances_file:
        distance_lines = [line.strip().split() for line in distances_file]
        distance_matrix = [[float(distance) for distance in line] for line in distance_lines]

    total_distance = 0
    total_time = 0
    total_gas = 0
    total_gas_cost = 0
    total_human_cost = 0
    num_cities = len(cities) - 1

    for i in range(num_cities - 1):
        start_city = cities[i]
        end_city = cities[i + 1]
        distance = distance_matrix[start_city][end_city]
        time = time_matrix[start_city][end_city]
        gas = gas_matrix[start_city][end_city]
        gas_cost = gas_cost_matrix[start_city][end_city]
        human_cost = human_cost_matrix[start_city][end_city]
        total_distance += distance
        total_time += time
        total_gas += gas
        total_gas_cost += gas_cost
        total_human_cost += human_cost

    hours = int(total_time)
    minutes = int((total_time - hours) * 60)

    print("Distance:", total_distance, "km")
    print("Time:", f"{hours} hours {minutes} minutes")
    print("Gas consumption:", total_gas, "L")
    print("Gas cost:", total_gas_cost, "€")
    print("Human cost:", total_human_cost, "€")

# Lancement de l'algorithme de recuit simulé
best_tour, best_distance = simulated_annealing(
    read_cost_matrix(cost_matrix_file_path), temp_init, cooling, temp_min, reheat_threshold, reheat_value,
    max_reheat_count)

# Calcul et affichage des résultats
calculate_results(best_tour)
