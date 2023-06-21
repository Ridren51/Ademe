import csv
import datetime
import os

import numpy as np
import psutil
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix

num_ants_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
alpha = 1
beta = 4
evaporation = 0.5
iterations = 100
nb_trucks = 1
perf_iterations = 1


def read_coordinates():
    coordinates = []
    with open('vendor/Coords/list.txt', 'r') as file:
        for line in file:
            parts = line.strip().split()
            index = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            coordinates.append((x, y))
    return coordinates


def read_distance_matrix(file_path):
    distance_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            distances = [float(distance) for distance in line.strip().split()]
            distance_matrix.append(distances)
    return np.array(distance_matrix)


def read_cost_matrix():
    cost_matrix = []
    with open('vendor/Coords/matrix.txt', 'r') as file:
        for line in file:
            row = [float(value) if value != 'nan' else np.nan for value in line.strip().split()]
            cost_matrix.append(row)
    return np.array(cost_matrix)


def choose_next_city(current_city, unvisited_cities, pheromone_matrix, distance_matrix, alpha, beta):
    probabilities = []
    total = 0

    for city in unvisited_cities:
        pheromone = pheromone_matrix[current_city][city] ** alpha
        distance = distance_matrix[current_city][city] ** beta

        if distance == 0 or np.isnan(distance):
            probabilities.append(0)
        else:
            probabilities.append(pheromone / distance)
            total += pheromone / distance

    if total == 0:
        return np.random.choice(unvisited_cities)

    probabilities = [p / total for p in probabilities]
    next_city_index = np.random.choice(range(len(unvisited_cities)), p=probabilities)
    return unvisited_cities[next_city_index]


def ant_colony(coordinates, distance_matrix, num_ants, alpha, beta, evaporation, nb_trucks):
    num_cities = len(coordinates)

    if nb_trucks == 1:
        best_path = None
        best_cost = float('inf')
        pheromone_matrix = np.ones((num_cities, num_cities)) * evaporation

        pbar = tqdm(total=iterations, desc="Running Ant Colony Optimization")
        paths = []
        costs = []
        distances = []

        for _ in range(iterations):
            pbar.update(1)

            for _ in range(num_ants):
                current_city = np.random.randint(0, num_cities)
                unvisited_cities = list(range(num_cities))
                unvisited_cities.remove(current_city)
                path = [current_city]
                cost = 0

                while unvisited_cities:
                    next_city = choose_next_city(current_city, unvisited_cities, pheromone_matrix,
                                                 distance_matrix, alpha, beta)
                    path.append(next_city)
                    cost += distance_matrix[current_city][next_city]
                    unvisited_cities.remove(next_city)
                    current_city = next_city

                path.append(path[0])
                cost += distance_matrix[path[-2]][path[-1]]

                paths.append(path)
                costs.append(cost)
                distances.append(cost)

                if cost < best_cost:
                    best_path = path
                    best_cost = cost

            pheromone_matrix *= (1 - evaporation)

            for i in range(num_ants):
                for j in range(num_cities):
                    pheromone_matrix[paths[i][j]][paths[i][j + 1]] += 1 / costs[i]

        pbar.close()

        return best_path, best_cost, distances
    else:
        distance_matrix = read_distance_matrix('vendor/Coords/distances.txt')

        clusters = [[] for _ in range(nb_trucks)]
        num_cities = len(coordinates)

        assigned = set()

        for i in range(num_cities):
            if i not in assigned:
                nearest_city = np.argmin(distance_matrix[i])
                cluster_id = len(assigned) % nb_trucks
                clusters[cluster_id].append(i)
                assigned.add(i)
                assigned.add(nearest_city)

        all_truck_paths = []
        total_cost = 0
        distances = []

        for cluster_id in range(nb_trucks):
            cluster_coordinates = [coordinates[i] for i in clusters[cluster_id]]
            cluster_distance_matrix = distance_matrix[clusters[cluster_id]][:, clusters[cluster_id]]

            best_path, best_cost, cluster_distances = ant_colony(cluster_coordinates, cluster_distance_matrix,
                                                                 num_ants, alpha, beta, evaporation, nb_trucks=1)

            all_truck_paths.append(best_path)
            total_cost += best_cost
            distances += cluster_distances

        return all_truck_paths, total_cost, distances


def plot_distance_iterations(distances):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Distance')
    ax1.set_title('Distance vs Iterations for different Num Ants values')

    for num_ants in num_ants_values:
        if nb_trucks == 1:
            best_path, best_cost, num_ants_distances = ant_colony(coordinates, distance_matrix, num_ants, alpha, beta,
                                                                  evaporation, nb_trucks)
        else:
            all_truck_paths, total_cost, num_ants_distances = ant_colony(coordinates, distance_matrix, num_ants, alpha,
                                                                         beta, evaporation, nb_trucks)

        ax1.plot(range(iterations), num_ants_distances, label=f"Num Ants = {num_ants}")

    ax1.legend()

    ax2.plot(timestamps, cpu_percentages, color='blue')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('CPU Usage (%)')
    ax2.set_title('CPU Usage')

    fig.tight_layout()
    plt.show()


def plot_distance_num_ants(distances):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel('Num Ants')
    ax.set_ylabel('Distance')
    ax.set_title('Distance vs Num Ants for fixed Alpha = 1.0 and Beta = 4.0')

    num_ants_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    avg_distances = []

    for num_ants in num_ants_values:
        if nb_trucks == 1:
            best_path, best_cost, num_ants_distances = ant_colony(coordinates, distance_matrix, num_ants, alpha, beta,
                                                                  evaporation, nb_trucks)
        else:
            all_truck_paths, total_cost, num_ants_distances = ant_colony(coordinates, distance_matrix, num_ants, alpha,
                                                                         beta, evaporation, nb_trucks)

        avg_distance = np.mean(num_ants_distances)
        avg_distances.append(avg_distance)

    ax.plot(num_ants_values, avg_distances)

    plt.tight_layout()
    plt.show()


cpu_percentages = []
memory_usages = []
timestamps = []
distances = []

coordinates = read_coordinates()
distance_matrix = read_cost_matrix()
num_cities = len(coordinates)

for i in range(perf_iterations):
    start_time = time.time()
    start_cpu_time = psutil.Process().cpu_times().user
    start_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024

    for num_ants in num_ants_values:
        if nb_trucks == 1:
            best_path, best_cost, num_ants_distances = ant_colony(coordinates, distance_matrix, num_ants, alpha, beta,
                                                                  evaporation, nb_trucks)
        else:
            all_truck_paths, total_cost, num_ants_distances = ant_colony(coordinates, distance_matrix, num_ants, alpha,
                                                                         beta, evaporation, nb_trucks)

        distances += num_ants_distances

        end_cpu_time = psutil.Process().cpu_times().user
        end_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024

        execution_time = time.time() - start_time
        cpu_time = end_cpu_time - start_cpu_time
        memory_usage = end_memory_usage - start_memory_usage

        cpu_percentages.append(psutil.cpu_percent())
        memory_usages.append(psutil.Process().memory_info().rss / 1024 / 1024)
        timestamps.append(int(execution_time))

plot_distance_iterations(distances)
plot_distance_num_ants(distances)
