import numpy as np
import psutil
import time

num_ants = 10
alpha = 1
beta = 2
evaporation = 0.5


def read_coordinates():
    coordinates = []
    with open('vendor/Coords/list.txt', 'r') as file:  # Read coordinates from 'list.txt' file
        for line in file:
            parts = line.strip().split()
            index = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            coordinates.append((x, y))
    return coordinates

def read_cost_matrix():
    cost_matrix = []
    with open('vendor/Coords/matrix.txt', 'r') as file:  # Read cost matrix from 'matrix.txt' file
        for line in file:
            row = [float(value) if value != 'nan' else np.nan for value in line.strip().split()]
            cost_matrix.append(row)
    return np.array(cost_matrix)

def choose_next_city(current_city, unvisited_cities, pheromone_matrix, distance_matrix, alpha, beta):
    probabilities = []
    total = 0

    for city in unvisited_cities:
        pheromone = pheromone_matrix[current_city][city] ** alpha  # Calculate pheromone value
        distance = 1 / distance_matrix[current_city][city] ** beta  # Calculate distance value
        probabilities.append(pheromone * distance)
        total += pheromone * distance

    probabilities = [p / total for p in probabilities]  # Calculate probabilities
    next_city_index = np.random.choice(range(len(unvisited_cities)), p=probabilities)  # Randomly select next city
    return unvisited_cities[next_city_index]

def ant_colony(coordinates, distance_matrix, num_ants, alpha, beta, evaporation):
    num_cities = len(coordinates)
    best_path = None
    best_cost = float('inf')
    pheromone_matrix = np.ones((num_cities, num_cities)) * evaporation  # Initialize pheromone matrix

    for _ in range(100):  # Run ant colony optimization for a fixed number of iterations
        paths = []
        costs = []

        for _ in range(num_ants):  # Create ant agents
            current_city = np.random.randint(0, num_cities)  # Choose random starting city
            unvisited_cities = list(range(num_cities))
            unvisited_cities.remove(current_city)
            path = [current_city]
            cost = 0

            while unvisited_cities:  # Construct path by iteratively choosing next city
                next_city = choose_next_city(current_city, unvisited_cities, pheromone_matrix,
                                             distance_matrix, alpha, beta)
                path.append(next_city)
                cost += distance_matrix[current_city][next_city]
                unvisited_cities.remove(next_city)
                current_city = next_city

            path.append(path[0])  # Complete the path by returning to the starting city
            cost += distance_matrix[path[-2]][path[-1]]  # Add the distance from the last city to the starting city

            paths.append(path)
            costs.append(cost)

            if cost < best_cost:  # Update best path and cost if a better solution is found
                best_path = path
                best_cost = cost

        pheromone_matrix *= (1 - evaporation)  # Evaporate pheromone on all edges

        for i in range(num_ants):  # Update pheromone matrix based on constructed paths
            for j in range(num_cities):
                pheromone_matrix[paths[i][j]][paths[i][j+1]] += 1 / costs[i]  # Add pheromone on the edge

    return best_path, best_cost

def running():
    coordinates = read_coordinates()  # Read coordinates from file
    distance_matrix = read_cost_matrix()  # Read cost matrix from file
    num_cities = len(coordinates)

    start_time = time.time()
    start_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time before running the algorithm
    start_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes

    best_path, best_cost = ant_colony(coordinates, distance_matrix, num_ants, alpha, beta, evaporation)  # Run ant colony optimization

    end_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time after running the algorithm
    end_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes
    end_time = time.time()

    execution_time = end_time - start_time  # Calculate total execution time
    cpu_time = end_cpu_time - start_cpu_time  # Calculate CPU time
    memory_usage = end_memory_usage - start_memory_usage  # Calculate memory usage

    print("Best path:", best_path)  # Print the best path found
    print("Best cost:", best_cost)  # Print the cost of the best path
    print("Execution time:", execution_time, "seconds")  # Print the total execution time
    print("CPU time:", cpu_time, "seconds")  # Print the CPU time
    print("Memory usage:", memory_usage, "MB")  # Print the memory usage

running()