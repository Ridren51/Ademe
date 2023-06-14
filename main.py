import math
import time
from itertools import combinations, groupby
import random as rd
import numpy as np


class Edge:
    def __init__(self, node1: str, node2: str, weight: int):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight
        self.probability = 0
        self.pheromone = 0


class Node:
    def __init__(self, node_name):
        self.node_name = str(node_name)
        self.degree = 0
        self.neighbors = []


class Graph:
    def __init__(self):
        self.adjacencyMatrix = []
        self.nodes = {}
        self.edges = {}

    def get_edge(self, node_name_1, node_name_2):
        try:
            return self.edges[f"{node_name_1},{node_name_2}"]
        except KeyError:
            try:
                return self.edges[f"{node_name_2},{node_name_1}"]
            except KeyError:
                return None

    def add_node(self, node_name: str):
        self.nodes[node_name] = Node(node_name)

    def add_nodes_from_list(self, nodes: list):
        for i in nodes:
            self.add_node(str(i))

    def add_edge(self, node1: str, node2: str, weight: int):
        if node1 in self.nodes and node2 in self.nodes:
            for i in self.edges.keys():
                if node1 in i and node2 in i:
                    if node1 != node2:
                        return

                    if f"{node1},{node2}" in self.edges:
                        return
            self.nodes[node1].degree += 1
            self.nodes[node1].neighbors.append(node2)
            self.nodes[node2].degree += 1
            self.nodes[node2].neighbors.append(node1)
            self.edges[f"{node1},{node2}"] = Edge(node1, node2, weight)

    def add_edges_from_list(self, edges: list):
        for edge in edges:
            self.add_edge(edge[0], edge[1], edge[2])

    def remove_node(self, node_name: str):
        if node_name in self.nodes:
            del self.nodes[node_name]
            # remove all links to this node
            for edge in self.edges.copy().values():
                if edge.node1 == node_name or edge.node2 == node_name:
                    print(edge)
                    self.remove_edge(edge.node1, edge.node2)

    def remove_edge(self, node1: str, node2: str):
        if f"{node1},{node2}" in self.edges:
            del self.edges[f"{node1},{node2}"]
            if node1 in self.nodes:
                self.nodes[node1].degree -= 1
                self.nodes[node1].neighbors.remove(node2)
            if node2 in self.nodes:
                self.nodes[node2].degree -= 1
                self.nodes[node2].neighbors.remove(node1)

    def is_eulerian_path(self):
        return all(self.nodes[i].degree % 2 == 0 for i in self.nodes)

    def get_adjency_matrix(self):
        self.adjacency_matrix()
        return self.adjacencyMatrix

    def print_adjency_matrix(self):
        self.adjacency_matrix()
        for i in self.adjacencyMatrix:
            print(i)

    def adjacency_matrix(self):
        start_time = time.time()
        matrix = [[0] * len(self.nodes) for _ in range(
            len(self.nodes))]  # create matrix of 0s with size of nodes x nodes (len(nodes) x len(nodes)) (rows x columns)
        for edge in self.edges.values():  # for each edge in the graph
            # set the value of the matrix at the index of the nodes to the weight of the edge
            matrix[list(self.nodes.keys()).index(edge.node1)][list(self.nodes.keys()).index(edge.node2)] = edge.weight
            # do the symmetry
            matrix[list(self.nodes.keys()).index(edge.node2)][list(self.nodes.keys()).index(edge.node1)] = edge.weight

        self.adjacencyMatrix = matrix
        print("adjacency matrix generated in ", (time.time() - start_time)*1000, "ms")

    def generate_random_graph(self, nodes: int=20): #todo coef pour chaque variable (conso, temps, cout)
        import networkx as nx

        start_time = time.time()

        p=.0001
        consumption_from_speed = {30: 55, 40: 48, 50: 44, 70: 33, 90: 38, 110: 44, 130: 51} # {speed in km/h: consumption in L/100km}
        job_cost_per_hour = 9  # €/h
        fuel_cost_per_liter = 1.5  # €/L

        def create_travel_cost():

            distance = rd.randint(1, 700)
            speed = rd.choice(list(consumption_from_speed.keys()))  # classic speeds limits in km/h
            consumption_per_hundred_km = consumption_from_speed[speed]  # consumption in L/100km

            travel_time = distance / speed
            consumption = consumption_per_hundred_km * (distance / 100)

            travel_time_cost = travel_time * job_cost_per_hour
            fuel_cost = consumption * fuel_cost_per_liter

            return math.floor(travel_time_cost + fuel_cost)  # total cost in €

        """
        Generates a random undirected graph, similarly to an Erdős-Rényi
        graph, but enforcing that the resulting graph is conneted
        """
        edges = combinations(range(nodes), 2)
        self.add_nodes_from_list(list(range(nodes)))

        if p >= 1:
            g = nx.complete_graph(nodes)
            self.node_and_edges_from_adjacency_matrix(nx.adjacency_matrix(g).todense())
        for node, node_edges in groupby(edges, key=lambda x: x[0]):
            node_edges = list(node_edges)
            while self.nodes[str(node)].degree < 2:
                choice = rd.choice(node_edges)
                self.add_edge(str(choice[0]), str(choice[1]), create_travel_cost())
            for e in node_edges:
                if rd.random() < p:
                    if e[0] != e[1]: #if same city then no distance nor speed
                        self.add_edge(str(e[0]), str(e[1]), create_travel_cost())
                    else:
                        self.add_edge(str(e[0]), str(e[1]), 0)
            if self.nodes[str(node)].degree < 2:
                self.add_edge(str(node), rd.choice(list(self.nodes.keys())), create_travel_cost())
        if self.nodes[str(nodes-1)].degree < 2:
            #fixes the case where the last node is not connected to the graph, since it is not iterated over
            self.add_edge(str(nodes-1), rd.choice(list(self.nodes.keys())), create_travel_cost())

        print("graph generated in ", (time.time() - start_time)*1000, "ms")


    def node_and_edges_from_adjacency_matrix(self, adjacency_matrix: list):
        for i in range(len(adjacency_matrix)):
            print(i)
            self.add_node(str(i))
        for i in range(len(adjacency_matrix)):
            print(f"{math.floor(i / len(adjacency_matrix) * 100)}%")
            for j in range(len(adjacency_matrix[i])):
                if adjacency_matrix[i][j] != 0:
                    self.add_edge(str(i), str(j), adjacency_matrix[i][j])

    def plot_graph(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        start_time = time.time()
        self.adjacency_matrix()

        graph = nx.empty_graph()
        graph.add_nodes_from(self.nodes.keys())

        layout = nx.kamada_kawai_layout(graph)
        for edge in self.edges.values():
            graph.add_edge(edge.node1, edge.node2)

        if len(self.edges)<=10:
            edge_labels = dict([((edge.node1, edge.node2), f'{edge.weight}') for edge in self.edges.values()])
            nx.draw_networkx_edge_labels(graph, pos=layout, edge_labels=edge_labels)

        fig_size = len(self.nodes)/15 if (len(self.nodes)>100) else 10
        nodes_size = 1000 if (len(self.nodes)>100) else 500

        plt.figure(figsize=(fig_size, fig_size))
        nx.draw_networkx(graph, pos=layout, node_size=nodes_size)
        plt.savefig('graph.svg')
        plt.show()
        print("graph plotted in ", (time.time() - start_time)*1000, "ms")


    # def aco(self):
    #     import numpy as np
    #     import psutil
    #     import time
    #
    #     num_ants = 10
    #     alpha = 1
    #     beta = 2
    #     evaporation = 0.5
    #
    #     def choose_next_city(current_city, unvisited_cities, pheromone_matrix, distance_matrix,best_path, cul_de_sac, alpha, beta):
    #         temp_probabilities = []
    #         total = 0
    #         # print("current_city",current_city)
    #         # print("unvisited_cities",unvisited_cities)
    #         for city in unvisited_cities:
    #             if distance_matrix[current_city][city] == 0 or city in cul_de_sac:
    #                 print("cul_de_sac",city)
    #                 temp_probabilities.append(0)
    #             else:
    #                 pheromone = pheromone_matrix[current_city][city] ** alpha  # Calculate pheromone value
    #                 # print("pheromone",pheromone_matrix[current_city][city])
    #                 distance = 1 / distance_matrix[current_city][city] ** beta  # Calculate distance value
    #                 temp_probabilities.append(pheromone * distance)
    #                 total += pheromone * distance
    #
    #         probabilities = []
    #
    #         for p in temp_probabilities: # Calculate probabilities
    #             # print("probability",p)
    #             if p!=0:
    #                 probabilities.append(p / total)  # Normalize probabilities
    #                 # print("probability p/T",p / total)
    #             else: # report 0 probability for cities that are not connected
    #                 probabilities.append(0)
    #
    #
    #         # print("probabilities",len(probabilities),probabilities)
    #         # print("unvisited_cities",len(unvisited_cities),unvisited_cities)
    #         # print("best_path",len(best_path),best_path)
    #         # print(current_city, best_path)
    #         if sum(probabilities)==0: #if all probabilities are 0 then go to the last city
    #             # print("added to cul de sac",best_path[-1])
    #             print("proba 0")
    #             cul_de_sac.append(best_path[-1]) #add the last city to the cul_de_sac list
    #             unvisited_cities.extend(best_path[-2:]) #add the previous city to the unvisited_cities list
    #             print("previous path",best_path)
    #             # best_path.pop() #remove the last city from the best_path list
    #             return best_path[-2]
    #         next_city_index = np.random.choice(range(len(unvisited_cities)),
    #                                            p=probabilities)  # Randomly select next city
    #         # print("next_city_index",next_city_index)
    #         return unvisited_cities[next_city_index]
    #
    #     def ant_colony(coordinates, num_ants, alpha, beta, evaporation):
    #         print(coordinates)
    #         num_cities = len(coordinates)
    #         best_path = []
    #         best_cost = float('inf')
    #         pheromone_matrix = np.ones((num_cities, num_cities)) * evaporation  # Initialize pheromone matrix
    #
    #         for _ in range(100):  # Run ant colony optimization for a fixed number of iterations
    #             paths = []
    #             costs = []
    #
    #             for _ in range(num_ants):  # Create ant agents
    #                 current_city = np.random.randint(0, num_cities)  # Choose random starting city
    #                 unvisited_cities = list(range(num_cities))
    #                 unvisited_cities.remove(current_city)
    #                 path = [current_city]
    #                 cost = 0
    #                 cul_de_sac = []
    #                 while unvisited_cities:  # Construct path by iteratively choosing next city
    #
    #                     next_city = choose_next_city(current_city, unvisited_cities, pheromone_matrix,
    #                                                  self.adjacencyMatrix,path,cul_de_sac, alpha, beta)
    #                     print("cul de sac",cul_de_sac)
    #                     path.append(next_city)
    #                     cost += self.adjacencyMatrix[current_city][next_city]
    #                     if next_city != current_city:
    #                         print("path",path)
    #                         print("next_city",next_city)
    #                         print("unvisited_cities",unvisited_cities)
    #                         unvisited_cities.remove(next_city)
    #                     current_city = next_city
    #
    #                 path.append(path[0])  # Complete the path by returning to the starting city
    #                 cost += self.adjacencyMatrix[path[-2]][
    #                     path[-1]]  # Add the distance from the last city to the starting city
    #
    #                 paths.append(path)
    #                 costs.append(cost)
    #
    #                 if cost < best_cost:  # Update best path and cost if a better solution is found
    #                     best_path = path
    #                     best_cost = cost
    #
    #             pheromone_matrix *= (1 - evaporation)  # Evaporate pheromone on all edges
    #
    #             for i in range(num_ants):  # Update pheromone matrix based on constructed paths
    #                 for j in range(num_cities):
    #                     pheromone_matrix[paths[i][j]][paths[i][j + 1]] += 1 / costs[i]  # Add pheromone on the edge
    #
    #         return best_path, best_cost
    #
    #     def running():
    #         print("running")
    #         num_cities = len(self.adjacencyMatrix)
    #
    #         start_time = time.time()
    #         start_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time before running the algorithm
    #         start_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes
    #
    #         best_path, best_cost = ant_colony(self.adjacencyMatrix, num_ants, alpha, beta,
    #                                           evaporation)  # Run ant colony optimization
    #
    #         end_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time after running the algorithm
    #         end_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes
    #         end_time = time.time()
    #
    #         execution_time = end_time - start_time  # Calculate total execution time
    #         cpu_time = end_cpu_time - start_cpu_time  # Calculate CPU time
    #         memory_usage = end_memory_usage - start_memory_usage  # Calculate memory usage
    #
    #         print("Best path:", best_path)  # Print the best path found
    #         print("Best cost:", best_cost)  # Print the cost of the best path
    #         print("Execution time:", execution_time, "seconds")  # Print the total execution time
    #         print("CPU time:", cpu_time, "seconds")  # Print the CPU time
    #         print("Memory usage:", memory_usage, "MB")  # Print the memory usage
    #
    #     running()

    def aco_proto(self, start_node):
        num_ants = 10
        alpha = 1
        beta = 2
        pheromone_quantity = 1
        evaporation = 0.5
        alread_visited_penalty = 0.1

        start_node = str(start_node)
        self.adjacency_matrix()
        print("aco_proto", start_node)
        print("nodes", self.nodes)
        print("node", self.nodes[start_node])
        print("adjacencyMatrix", self.adjacencyMatrix)

        current_city = self.nodes[start_node]
        unvisited_cities = list(self.nodes.keys())
        path = []
        cost = 0
        cul_de_sac = []
        last_city = None
        print("condition", (unvisited_cities != [] or current_city.node_name == start_node))
        while unvisited_cities!=[] or current_city.node_name != start_node:
            print("condition",(unvisited_cities!=[] or current_city.node_name == start_node))

            neighbor_choice_probabilities = []

            total = 0

            # print("current_city", current_city.node_name)
            # print("last_city", last_city)
            # print("neighbors", current_city.neighbors)
            #choose next city
            for neighbor in current_city.neighbors:
                # print("neigbhours", neighbor)
                # print("distance", self.get_edge(current_city.node_name, neighbor).weight)
                # print("pheromone", self.get_edge(current_city.node_name, neighbor).pheromone)
                edge = self.get_edge(current_city.node_name, neighbor)
                pheromone = edge.pheromone ** alpha  # Calculate pheromone value
                # print("pheromone", pheromone)
                # print("pheromone",pheromone_matrix[current_city][city])
                distance = 1/edge.weight ** beta if edge.weight != 0 else 0
                score = edge.pheromone * distance if edge.pheromone != 0 else distance

                if not unvisited_cities and neighbor == start_node:
                    score = score * 100
                elif neighbor in path:
                    if neighbor == last_city.node_name:
                        score = 0
                    score = alread_visited_penalty * score # penalize already visited cities to avoid loops
                total += score
                neighbor_choice_probabilities.append(score)
            # print("neighbor_choice_probabilities", neighbor_choice_probabilities)
            probabilities = []
            for p in neighbor_choice_probabilities:
                if p == 0:
                    probabilities.append(0)
                    continue
                probabilities.append(p / total)  # Calculate probabilities
            # print("probabilities", sum(probabilities),probabilities)
            # print("node_name", type(current_city.node_name))
            # print("unvisited_cities", unvisited_cities)
            if current_city.node_name in unvisited_cities:
                unvisited_cities.remove(current_city.node_name)
            # print("unvisited_cities", unvisited_cities)
            path.append(current_city.node_name)
            last_city = current_city
            current_city = self.nodes[np.random.choice(current_city.neighbors, p=probabilities)]  # Choose next city
                    # print("next_city", current_city)
        path.append(start_node)
        print("condition",(unvisited_cities!=[] or current_city.node_name == start_node))
        print(unvisited_cities!=[])
        print(current_city.node_name == start_node)

        print("final path", path)





    def print_graph(self):
        print("Nodes:")
        for key, value in self.nodes.items():
            print(key)
            for i in self.edges.keys():
                if key in i:
                    print("  ", i, self.edges[i].weight)


graphe = Graph()
graphe.generate_random_graph(10)
# print(graphe.is_eulerian_path())
# graphe.print_adjency_matrix()
graphe.plot_graph()
graphe.print_adjency_matrix()

for i in graphe.nodes:
    print(graphe.nodes[i].neighbors)
# print(graphe.adjacencyMatrix)
# print(graphe.print_graph())
