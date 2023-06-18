import datetime
import math
import time
from itertools import combinations, groupby
import random as rd
import numpy as np
import psutil


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


class Utils:

    def instance_starter(self, instance=None, instance_size=-1):
        grapher = Graph()
        if instance is None:
            if instance_size != -1:
                grapher.generate_random_graph(instance_size)
            else:
                grapher.generate_random_graph(rd.randint(10, 100))

        else:
            grapher.node_and_edges_from_adjacency_matrix(instance)
        return grapher
    def performance_test_multiple_instances(self, func, func_params:dict, iterations:int=1, instance_size:int=-1): #wrapper for performance test
        """
        :param func: function to test
        :param func_params: default parameters of the function
        :param iterations: number of iterations to run the test
        :param instance_size: number of nodes in the graph
        """

        import csv
        import os
        print("Running performance test for ", func.__name__, " with ", iterations, " iterations and ", instance_size, " nodes")

        filename=f'vendor/benchmarks/{func.__name__}/'
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(f"{filename}/{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.csv", mode='w', newline='') as benchfile:
            writer = csv.writer(benchfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["iteration", "runtime (ms)", "CPU time (ms)", "memory (mb)", "nb_nodes", "nb_edges", "cost", "path"])

            for iteration in range(iterations):
                grapher=self.instance_starter(instance_size=instance_size)

                start_time = time.time()
                start_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time before running the algorithm
                start_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes


                result = func(graph=grapher,**func_params)
                print("result: ", result)


                end_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time after running the algorithm
                end_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes
                end_time = time.time()

                writer.writerow([iteration, (end_time - start_time)*1000, (end_cpu_time - start_cpu_time)*1000, end_memory_usage - start_memory_usage, len(grapher.nodes), len(grapher.edges), result[0],result[1]])
        benchfile.close()

    def performance_test(self, func, func_params:dict, iterations:int=1, instance_size:int=-1, instance: list = None): #wrapper for performance test

        """
        :param func: function to test
        :param func_params: default parameters of the function
        :param iterations: number of iterations to run the test
        :param instance_size: number of nodes in the graph
        :param instance: adjacency matrix of the graph
        """

        import os
        import csv

        print("Running performance test for ", func.__name__, " with ", iterations, " iterations and ",
              instance_size, " nodes")

        grapher = self.instance_starter(instance=instance, instance_size=instance_size)



        filename = f'vendor/benchmarks/{func.__name__}/'
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(f"{filename}/{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.csv", mode='w', newline='') as benchfile:
            writer = csv.writer(benchfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["iteration", "runtime (ms)", "CPU time (ms)", "memory (mb)", "nb_nodes", "nb_edges", "cost", "path"])

            for iteration in range(iterations):
                start_time = time.time()
                start_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time before running the algorithm
                start_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes

                result = func(graph=grapher, **func_params)
                print("result: ", result)

                end_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time after running the algorithm
                end_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes
                end_time = time.time()

                writer.writerow([iteration, (end_time - start_time)*1000, (end_cpu_time - start_cpu_time)*1000, end_memory_usage - start_memory_usage, len(grapher.nodes), len(grapher.edges), result[0], result[1]])
        benchfile.close()


    def aco_parameters_test(self, func, func_params:dict, parameters:dict, iterations:int=100, instance_size:int=-1, instance: list = None):    #wrapper which tests a function with varying parameters
        """
        :param func: function to test
        :param func_params: default parameters of the function
        :param parameters: dictionary of parameters to test with min and max values ex: {"param1": [min, max], "param2": [min, max]}
        :param iterations: number of iterations to run the test
        :param instance_size: number of nodes in the graph
        :param instance: adjacency matrix of the graph
        """
        import os
        import csv


        print("Running parameters test for aco with ", iterations, " iterations and ",
              instance_size, " nodes")

        #test all parameters combinations possible with the given parameters and their min and max values
        # for i, j, k in product(range(parameters['num_ants'][0], parameters['num_ants'][1]), range(parameters['alpha'][0], parameters['alpha'][1]), range(parameters['beta'][0], parameters['beta'][1]), range(parameters['iterations'][0], parameters['iterations'][1])):
        #     # for l in product(range(parameters['beta'][0], parameters['beta'][1]), range(parameters['evaporation'][0], parameters['evaporation'][1])):
        #     #     for m, n in product(range(parameters['already_visited_penalty'][0], parameters['already_visited_penalty'][1]), range(parameters['iterations'][0], parameters['iterations'][1])):
        #     #         self.performance_test(func, {'num_ants': i, 'alpha': j, 'beta': k, 'evaporation': l, 'already_visited_penalty': m, 'iterations': n}, iterations, instance_size, instance)
        #     print(i, j, k)
        grapher = self.instance_starter(instance=instance, instance_size=instance_size)
        grapher.plot_graph()
        filename = 'vendor/benchmarks/param_test/'
        os.makedirs(os.path.dirname(filename), exist_ok=True)



        with open(f"{filename}aco_{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.csv", mode='w', newline='') as benchfile:
            writer = csv.writer(benchfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["iteration", "runtime (ms)", "CPU time (ms)", "memory (mb)", "nb_nodes", "nb_edges", "parameters", "cost", "path"])

            for iteration in range(iterations):
                start_time = time.time()
                start_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time before running the algorithm
                start_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes

                func_params = {'num_ants': rd.randint(parameters['num_ants'][0],parameters['num_ants'][1]), 'alpha': rd.randint(parameters['alpha'][0],parameters['alpha'][1]), 'beta': rd.randint(parameters['beta'][0],parameters['beta'][1]), 'evaporation': rd.uniform(parameters['evaporation'][0],parameters['evaporation'][1]), 'already_visited_penalty': rd.uniform(parameters['already_visited_penalty'][0],parameters['already_visited_penalty'][1])}
                print(func_params)

                result = aco(graph=grapher, start_node=0, **func_params)

                end_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time after running the algorithm
                end_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes
                end_time = time.time()

                writer.writerow([iteration, (end_time - start_time) * 1000, (end_cpu_time - start_cpu_time) * 1000,
                                 end_memory_usage - start_memory_usage, len(grapher.nodes), len(grapher.edges), func_params, result[0],
                                 result[1]])
            benchfile.close()








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

    def clear(self):
        self.nodes = {}
        self.edges = {}
        self.adjacencyMatrix = []

    def generate_random_graph(self, nodes: int=20): #todo coef pour chaque variable (conso, temps, cout)
        import networkx as nx

        #clear existing graph
        self.clear()

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
            choice = rd.choice(node_edges)
            self.add_edge(str(choice[0]), str(choice[1]), create_travel_cost())

            for e in node_edges:
                if rd.random() < p:
                    if e[0] != e[1]: #if same city then no distance nor speed
                        self.add_edge(str(e[0]), str(e[1]), create_travel_cost())
                    else:
                        self.add_edge(str(e[0]), str(e[1]), 0)

        print("graph generated in ", (time.time() - start_time)*1000, "ms")


    def node_and_edges_from_adjacency_matrix(self, adjacency_matrix: list):
        # clear existing graph
        self.clear()
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


    def print_graph(self):
        print("Nodes:")
        for key, value in self.nodes.items():
            print(key)
            for i in self.edges.keys():
                if key in i:
                    print("  ", i, self.edges[i].weight)


def aco(graph:Graph, start_node, num_ants:int = 10, alpha:int = 1, beta:int = 2, evaporation:float = 0.5, already_visited_penalty:float = 0.5, iterations:int = 10):
    """
    :param graph: graph to use
    :param start_node: node to start from
    :param num_ants: number of ants to use
    :param alpha: pheromone importance
    :param beta: distance importance
    :param evaporation: pheromone evaporation rate
    :param already_visited_penalty: penalty for already visited nodes
    :param iterations: number of iterations to run
    :return: tuple of (best path cost, best path)
    """


    best_path = []

    start_node = str(start_node)
    # print("aco_proto", start_node)
    # print("nodes", self.nodes)
    # print("node", self.nodes[start_node])

    start_time = time.time()

    for _ in range(iterations): # Run ant colony optimization for a fixed number of iterations
        paths = []
        for _ in range(num_ants): # Create ant agents
            print("ant", _)
            current_city = graph.nodes[start_node]
            unvisited_cities = list(graph.nodes.keys())
            path = []
            edges = []
            cost = 0
            last_city = None

            iteration_time = time.time()

            while (unvisited_cities!=[] or current_city.node_name != start_node): #and time.time()-iteration_time < 10: # Construct path by iteratively choosing next city until all cities have been visited or time is up (10 seconds)
                # print("time", time.time()-iteration_time) #fixme
                neighbor_choice_probabilities = []
                total = 0

                #choose next city
                for neighbor in current_city.neighbors:

                    edge = graph.get_edge(current_city.node_name, neighbor)
                    pheromone = edge.pheromone ** alpha  # Calculate pheromone value
                    distance = 1/edge.weight ** beta if edge.weight != 0 else 0
                    score = edge.pheromone * distance if edge.pheromone != 0 else 1*distance

                    if not unvisited_cities and neighbor == start_node:
                        score = score * 1000
                    elif neighbor in path:
                        if neighbor == last_city.node_name:
                            score = score/1000
                        score = already_visited_penalty * score # penalize already visited cities to avoid loops
                    total += score
                    neighbor_choice_probabilities.append(score)

                probabilities = []
                for p in neighbor_choice_probabilities:
                    if p == 0:
                        probabilities.append(0)
                        continue
                    probabilities.append(p / total)  # Calculate probabilities
                if current_city.node_name in unvisited_cities:
                    unvisited_cities.remove(current_city.node_name)
                path.append(current_city.node_name)
                # print("current_city", unvisited_cities)

                last_city = current_city
                # print("probabilities", sum(probabilities),probabilities)
                # print("path", path)
                current_city = graph.nodes[np.random.choice(current_city.neighbors, p=probabilities)]  # Choose next city
                edges.append(graph.get_edge(last_city.node_name, current_city.node_name))
                cost += graph.get_edge(last_city.node_name, current_city.node_name).weight

            path.append(start_node)
            cost += graph.get_edge(last_city.node_name, start_node).weight
            paths.append((cost,path))

            for edge in list(set(edges)):
                print("edge", edge.node1, edge.node2)
                edge.pheromone += 1 / cost
                edge.pheromone *= (1 - evaporation)  # Evaporate pheromone on all edges

        best_path = min(paths, key=lambda x: x[0])
    print("time", (time.time()-start_time)*1000,"ms")
    print("best cost", best_path)
    return best_path


utils = Utils()

utils.performance_test(aco, {'start_node' : 0}, 10, 10)
