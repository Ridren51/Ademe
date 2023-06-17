import numpy as np
import psutil
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix

num_ants = 20
alpha = 1
beta = 2
evaporation = 0.5
iterations = 100
nb_trucks = 2

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
distance_matrix = read_distance_matrix('vendor/Coords/distances.txt')

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

        # Avoid division by zero or NaN values
        if distance == 0 or np.isnan(distance):
            probabilities.append(0)
        else:
            probabilities.append(pheromone / distance)
            total += pheromone / distance

    if total == 0:
        # If total is still zero, choose a random next city
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

        # Wrapping range with tqdm to create a progress bar
        for _ in tqdm(range(iterations), desc="Running Ant Colony Optimization"):
            paths = []
            costs = []

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

                if cost < best_cost:
                    best_path = path
                    best_cost = cost

            pheromone_matrix *= (1 - evaporation)

            for i in range(num_ants):
                for j in range(num_cities):
                    pheromone_matrix[paths[i][j]][paths[i][j+1]] += 1 / costs[i]

        return best_path, best_cost
    else:
        # Cas pour plusieurs camions avec clustering
        distance_matrix = read_distance_matrix('vendor/Coords/distances.txt')

        # Création des clusters
        clusters = [[] for _ in range(nb_trucks)]
        num_cities = len(coordinates)

        assigned = set()  # Villes déjà assignées à un cluster

        for i in range(num_cities):
            if i not in assigned:
                nearest_city = np.argmin(distance_matrix[i])  # Indice de la ville la plus proche
                cluster_id = len(assigned) % nb_trucks  # Identifiant du cluster
                clusters[cluster_id].append(i)  # Ajouter la ville au cluster
                assigned.add(i)  # Marquer la ville comme assignée
                assigned.add(nearest_city)  # Marquer la ville la plus proche comme assignée

        # Storing paths for all trucks
        all_truck_paths = []
        total_cost = 0

        # Running ant colony optimization for each cluster
        for cluster_id in range(nb_trucks):
            cluster_coordinates = [coordinates[i] for i in clusters[cluster_id]]
            cluster_distance_matrix = distance_matrix[clusters[cluster_id]][:, clusters[cluster_id]]

            # Running ant colony optimization on the cluster
            best_path, best_cost = ant_colony(cluster_coordinates, cluster_distance_matrix, num_ants, alpha, beta,
                                              evaporation, nb_trucks=1)

            # Adding paths and costs for each truck
            all_truck_paths.append(best_path)
            total_cost += best_cost

        # Return all the paths and the total cost
        return all_truck_paths, total_cost
def running():
    # Plot the CPU usage graph
    cpu_percentages = []
    memory_usages = []
    timestamps = []

    # Read coordinates and distance matrix
    coordinates = read_coordinates()
    distance_matrix = read_cost_matrix()
    num_cities = len(coordinates)

    # Start time and resource usage
    start_time = time.time()
    start_cpu_time = psutil.Process().cpu_times().user
    start_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024

    # Run the ant colony optimization algorithm
    if nb_trucks == 1:
        best_path, best_cost = ant_colony(coordinates, distance_matrix, num_ants, alpha, beta, evaporation, nb_trucks)
    else:
        all_truck_paths, total_cost = ant_colony(coordinates, distance_matrix, num_ants, alpha, beta, evaporation,
                                                 nb_trucks)
    # End time and resource usage
    end_cpu_time = psutil.Process().cpu_times().user
    end_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
    end_time = time.time()

    # Calculate execution time and resource usage
    execution_time = end_time - start_time
    cpu_time = end_cpu_time - start_cpu_time
    memory_usage = end_memory_usage - start_memory_usage

    # Print results
    if nb_trucks == 1:
        print("Best path:", best_path)
        print("Best cost:", best_cost)
        np.savetxt('vendor/Coords_ant/road.txt', best_path, fmt='%.0f')
    else:
        print("Best paths:", all_truck_paths)
        print("Total cost:", total_cost)
        with open('vendor/Coords_ant/composite_road.txt', 'w') as file:
            for i, truck_path in enumerate(all_truck_paths):
                for city in truck_path:
                    file.write(str(city) + '\n')
                if i < len(all_truck_paths) - 1:
                    file.write('-----' + '\n')
    print("Execution time:", execution_time, "seconds")
    print("CPU time:", cpu_time, "seconds")
    print("Memory usage:", memory_usage, "MB")


    # Plot CPU and memory usage in real-time
    for i in range(int(execution_time) + 1):
        cpu_percentages.append(psutil.cpu_percent())
        memory_usages.append(psutil.Process().memory_info().rss / 1024 / 1024)
        timestamps.append(i)

        # Plot CPU usage graph
        plt.subplot(211)
        plt.plot(timestamps, cpu_percentages, color='blue')
        plt.xlabel('Time (seconds)')
        plt.ylabel('CPU Usage (%)')
        plt.title('CPU Usage')

        # Plot memory usage graph
        plt.subplot(212)
        plt.plot(timestamps, memory_usages, color='red')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage')
        plt.draw()
    plt.show()

def calculate_results():
    # Path of the file containing the list of cities
    cities_file_path = 'vendor/Coords_ant/road.txt'
    composite_cities_file_path = 'vendor/Coords_ant/composite_road.txt'
    # Path of the file containing the distance matrix
    distances_file_path = 'vendor/Coords/distances.txt'
    times_file_path = 'vendor/Coords/times.txt'
    gas_file_path = 'vendor/Coords/gas.txt'
    gas_cost_file_path = 'vendor/Coords/gas_cost.txt'
    human_cost_file_path = 'vendor/Coords/human_cost.txt'

    # Read the list of cities from the file
    with open(cities_file_path, 'r') as cities_file:
        cities = [int(line.strip()) for line in cities_file]

    # Read the distance matrix from the file
    with open(distances_file_path, 'r') as distances_file:
        distance_lines = [line.strip().split() for line in distances_file]
        distance_matrix = [[float(distance) for distance in line] for line in distance_lines]

    # Read the time matrix from the file
    with open(times_file_path, 'r') as times_file:
        time_lines = [line.strip().split() for line in times_file]
        time_matrix = [[float(time) for time in line] for line in time_lines]

    # Read the gas consumption matrix from the file
    with open(gas_file_path, 'r') as gas_file:
        gas_lines = [line.strip().split() for line in gas_file]
        gas_matrix = [[float(gas) for gas in line] for line in gas_lines]

    # Read the gas cost matrix from the file
    with open(gas_cost_file_path, 'r') as gas_cost_file:
        gas_cost_lines = [line.strip().split() for line in gas_cost_file]
        gas_cost_matrix = [[float(gas_cost) for gas_cost in line] for line in gas_cost_lines]

    # Read the human cost matrix from the file
    with open(human_cost_file_path, 'r') as human_cost_file:
        human_cost_lines = [line.strip().split() for line in human_cost_file]
        human_cost_matrix = [[float(human) for human in line] for line in human_cost_lines]

    if nb_trucks == 1:
        # Calculate the total distance of the circuit
        total_distance = 0
        total_time = 0
        total_gas = 0
        total_gas_cost = 0
        total_human_cost = 0
        num_cities = len(cities)
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
            hours = int(total_time)  # Integer part of hours
            difference = total_time - hours  # Difference between the approximation and the integer part
            minutes = int(difference * 60)  # Conversion of the difference to minutes

        print("Distance:", total_distance, "km")
        print("Time:", f"{hours} hours {minutes} minutes")
        print("Gas consumption:", total_gas, "L")
        print("Gas cost:", total_gas_cost, "€")
        print("Human cost:", total_human_cost, "€")
    else:
        # Cas pour plusieurs camions
        with open(composite_cities_file_path, 'r') as composite_cities_file:
            lines = composite_cities_file.readlines()
            truck_paths = []  # Liste pour stocker les chemins de chaque camion
            current_path = []  # Liste temporaire pour stocker le chemin courant
            for line in lines:
                if line.strip() != '-----':
                    current_path.append(int(line.strip()))
                else:
                    truck_paths.append(current_path)
                    current_path = []
            if current_path:  # Ajouter le dernier chemin s'il n'est pas vide
                truck_paths.append(current_path)

        # Calculer la distance totale du circuit pour chaque camion
        for truck_index, cities in enumerate(truck_paths):
            total_distance = 0
            total_time = 0
            total_gas = 0
            total_gas_cost = 0
            total_human_cost = 0
            num_cities = len(cities)
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

            hours = int(total_time)  # Partie entière des heures
            difference = total_time - hours  # Différence entre l'approximation et la partie entière
            minutes = int(difference * 60)  # Conversion de la différence en minutes

            print(f"Truck {truck_index + 1}:")
            print("  Distance:", total_distance, "km")
            print("  Time:", f"{hours} hours {minutes} minutes")
            print("  Gas consumption:", total_gas, "L")
            print("  Gas cost:", total_gas_cost, "€")
            print("  Human cost:", total_human_cost, "€")
            print()
running()
calculate_results()
