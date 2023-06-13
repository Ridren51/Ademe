import math
import time
from itertools import combinations, groupby
import random as rd


class Edge:
    def __init__(self, node1: str, node2: str, weight: int):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight


class Node:
    def __init__(self, node_name: str):
        self.nodeName = node_name
        self.degree = 0
        self.neighbors = []


class Graph:
    def __init__(self):
        self.adjacencyMatrix = []
        self.nodes = {}
        self.edges = {}

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

    def generate_random_graph(self, nodes: int=20):
        import networkx as nx

        start_time = time.time()

        p=.00001
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

    def generate_random_graph(self, nodes: int=20):
        import networkx as nx
        import random as rd
        start_time = time.time()
        p=.00001
        consumption_from_speed = {30: 55, 40: 48, 50: 44, 70: 33, 90: 38, 110: 44, 130: 51} # {speed in km/h: consumption in L/100km}
        job_cost_per_hour = 9  # €/h
        fuel_cost_per_liter = 1.5  # €/L

        """
        Generates a random undirected graph, similarly to an Erdős-Rényi
        graph, but enforcing that the resulting graph is conneted
        """
        edges = combinations(range(nodes), 2)
        self.add_nodes_from_list(list(range(nodes)))

        if p >= 1:
            g = nx.complete_graph(nodes)
            self.node_and_edges_from_adjacency_matrix(nx.adjacency_matrix(g).todense())
        for _, node_edges in groupby(edges, key=lambda x: x[0]):

            node_edges = list(node_edges)
            self.add_edge(str(rd.choice(node_edges)[0]), str(rd.choice(node_edges)[1]), create_travel_cost())
            self.add_edge(str(rd.choice(node_edges)[0]), str(rd.choice(node_edges)[1]), create_travel_cost())

            for e in node_edges:
                if rd.random() < p:
                    if e[0] != e[1]: #if same city then no distance nor speed
                        self.add_edge(str(e[0]), str(e[1]), create_travel_cost())
                    else:
                        self.add_edge(str(e[0]), str(e[1]), 0)

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

        layout = nx.random_layout(graph)
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
