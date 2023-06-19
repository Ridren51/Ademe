import math
import time
import random as rd
from itertools import combinations, groupby

class Node:
    def __init__(self, node_name):
        self.node_name = str(node_name)
        self.degree = 0
        self.neighbors = []

class Edge:
    def __init__(self, node1: str, node2: str, weight: int):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight
        self.pheromone = 0

class Graph:
    def __init__(self):
        self.adjacencyMatrix = [] # Adjacency matrix
        self.nodes = {} # dict of {node_name: Node object}
        self.edges = {} # dict of {f"{node1},{node2}": Edge object}

    def get_edge(self, node_name_1, node_name_2):
        """
        Returns edge object if edge exists, else returns None
        :param node_name_1: node1 name
        :param node_name_2: node2 name
        :return: Edge object or None
        """

        try:
            return self.edges[f"{node_name_1},{node_name_2}"] # Try to get edge whit order node1, node2
        except KeyError:
            try:
                return self.edges[f"{node_name_2},{node_name_1}"] # Try to get edge whit order node2, node1
            except KeyError:
                return None # Edge does not exist

    def add_node(self, node_name: str):
        """
        Adds node to graph
        :param node_name: name of node
        :return:
        """

        self.nodes[node_name] = Node(node_name)

    def add_nodes_from_list(self, nodes: list):
        """
        Adds nodes to graph from list
        :param nodes: nodes [node1, node2, ...]
        :return: None
        """

        for i in nodes:
            self.add_node(str(i))

    def add_edge(self, node1: str, node2: str, weight: int):
        """
        Adds edge to graph
        :param node1: node1 name
        :param node2: node2 name
        :param weight: weight of edge
        :return: None
        """

        if node1 in self.nodes and node2 in self.nodes: # If both nodes exist
            for i in self.edges.keys(): # Check if edge already exists
                if node1 in i and node2 in i: # If edge already exists
                    if node1 != node2: # If edge is not a loop
                        return

                    if f"{node1},{node2}" in self.edges: # If edge exists with order node1, node2
                        return
            self.nodes[node1].degree += 1 # Increase degree of nodes
            self.nodes[node1].neighbors.append(node2) # Add node2 to node1 neighbors
            self.nodes[node2].degree += 1
            self.nodes[node2].neighbors.append(node1)
            self.edges[f"{node1},{node2}"] = Edge(node1, node2, weight) # Add edge to graph

    def add_edges_from_list(self, edges: list):
        """
        Adds edges to graph from list
        :param edges: edges [[node1, node2, weight], [node1, node2, weight], ...]
        :return None
        """

        for edge in edges:
            self.add_edge(edge[0], edge[1], edge[2])

    def remove_node(self, node_name: str):
        """
        Removes node from graph (if node exists)
        :param node_name: node name
        :return: None
        """

        if node_name in self.nodes:
            del self.nodes[node_name]

            # remove all links to this node
            for edge in self.edges.copy().values(): # copy() to avoid RuntimeError: dictionary changed size during iteration
                if edge.node1 == node_name or edge.node2 == node_name:
                    print(edge)
                    self.remove_edge(edge.node1, edge.node2)

    def remove_edge(self, node1: str, node2: str): # Removes edge from graph
        """
        Removes edge from graph (if edge exists)(order does not matter)
        :param node1: first node name
        :param node2: second node name
        :return: None
        """

        if f"{node1},{node2}" in self.edges: # If edge exists with order node1, node2
            del self.edges[f"{node1},{node2}"]
            if node1 in self.nodes:
                self.nodes[node1].degree -= 1
                self.nodes[node1].neighbors.remove(node2)
            if node2 in self.nodes:
                self.nodes[node2].degree -= 1
                self.nodes[node2].neighbors.remove(node1)

        elif f"{node2},{node1}" in self.edges: # If edge exists with order node2, node1
            del self.edges[f"{node2},{node1}"]
            if node1 in self.nodes:
                self.nodes[node1].degree -= 1
                self.nodes[node1].neighbors.remove(node2)
            if node2 in self.nodes:
                self.nodes[node2].degree -= 1
                self.nodes[node2].neighbors.remove(node1)

    def is_eulerian_path(self):
        """
        Checks if graph has an eulerian path (all nodes have even degree except 2 nodes that have odd degree)
        :return: True if graph has an eulerian path, else returns False
        """

        return all(self.nodes[i].degree % 2 == 0 for i in self.nodes)

    def get_adjency_matrix(self):
        """
        Generates adjacency matrix
        :return: list of lists (matrix)
        """

        self.adjacency_matrix()
        return self.adjacencyMatrix

    def print_adjency_matrix(self):
        """
        Prints formatted adjacency matrix
        :return: None
        """

        self.adjacency_matrix()
        for i in self.adjacencyMatrix:
            print(i)

    def adjacency_matrix(self):
        """
        Generates adjacency matrix
        :return: None
        """

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
        """
        Clears graph
        :return: None
        """

        self.nodes = {}
        self.edges = {}
        self.adjacencyMatrix = []

    def create_travel_cost(self, consumption_from_speed=None, job_cost_per_hour = 9, fuel_cost_per_liter = 1.5, distance = rd.randint(1, 100)):
        # {speed in km/h: consumption in L/100km}
        # job_cost_per_hour €/h
        # fuel_cost_per_liter €/L
        if consumption_from_speed is None:
            consumption_from_speed = {
                30: 55,
                40: 48,
                50: 44,
                70: 33,
                90: 38,
                110: 44,
                130: 51,
            }


        speed = rd.choice(list(consumption_from_speed.keys()))  # classic speeds limits in km/h
        consumption_per_hundred_km = consumption_from_speed[speed]  # consumption in L/100km

        travel_time = distance / speed  # travel time in hours
        consumption = consumption_per_hundred_km * (distance / 100)  # consumption in L

        travel_time_cost = travel_time * job_cost_per_hour  # total job cost in €
        fuel_cost = consumption * fuel_cost_per_liter  # total fuel cost in €

        return math.floor(travel_time_cost + fuel_cost)  # total cost in €

    def generate_random_graph(self, nodes: int=20, p:float=.5):
        """
        Generates random graph with nodes number and probability of having an edge
        :param nodes: number of nodes
        :param p: probability of link between two nodes
        :return: None
        """

        #todo coef pour chaque variable (conso, temps, cout)


        self.clear() #clear existing graph

        start_time = time.time()





        """
        Generates a random undirected graph, similarly to an Erdős-Rényi
        graph, but enforcing that the resulting graph is conneted
        """
        edges = combinations(range(nodes), 2) # all possible edges
        self.add_nodes_from_list(list(range(nodes))) # add nodes to graph (from 0 to nodes-1)

        if p >= 1: # if p >= 1 then all nodes are connected
            for node_edges in edges:
                node_edge=list(node_edges)
                node_edge.append(self.create_travel_cost())
                self.add_edge(str(node_edge[0]), str(node_edge[1]), node_edge[2])

        for node, node_edges in groupby(edges, key=lambda x: x[0]): # for each node and its edges
            node_edges = list(node_edges)
            choice = rd.choice(node_edges) # choose a random edge
            self.add_edge(str(choice[0]), str(choice[1]), self.create_travel_cost()) # add edge to graph

            for e in node_edges:
                if rd.random() < p: # randomize if edge is added to graph
                    if e[0] != e[1]: #if the same city then no distance nor speed
                        self.add_edge(str(e[0]), str(e[1]), self.create_travel_cost()) # add edge to graph
                    else:
                        self.add_edge(str(e[0]), str(e[1]), 0) # add edge to graph

        self.adjacency_matrix() # generate adjacency matrix
        print("graph generated in ", (time.time() - start_time)*1000, "ms")


    def node_and_edges_from_adjacency_matrix(self, adjacency_matrix: list):
        """
        Generates nodes and edges from adjacency matrix
        :param adjacency_matrix: adjacency matrix [[0, 1, 2], [1, 0, 3], [2, 3, 0], ...]
        :return: None
        """

        self.clear() # clear existing graph
        for i in range(len(adjacency_matrix)): # create nodes
            self.add_node(str(i))
        for i in range(len(adjacency_matrix)): # create edges
            for j in range(len(adjacency_matrix[i])):
                if adjacency_matrix[i][j] != 0: # if there is a link between the two nodes
                    self.add_edge(str(i), str(j), adjacency_matrix[i][j])

    def plot_graph(self):
        """
        Plots graph using matplotlib and networkx
        :return: None
        """

        import networkx as nx
        import matplotlib.pyplot as plt
        start_time = time.time()
        self.adjacency_matrix()

        graph = nx.empty_graph()
        graph.add_nodes_from(self.nodes.keys())

        layout = nx.kamada_kawai_layout(graph) # layout of the graph

        for edge in self.edges.values():
            graph.add_edge(edge.node1, edge.node2) # add edges to the graph

        if len(self.nodes)<=10:
            edge_labels = dict([((edge.node1, edge.node2), f'{edge.weight}') for edge in self.edges.values()]) # add edge labels
            nx.draw_networkx_edge_labels(graph, pos=layout, edge_labels=edge_labels)

        fig_size = len(self.nodes)/15 if (len(self.nodes)>100) else 10
        nodes_size = 1000 if (len(self.nodes)>100) else 500

        plt.figure(figsize=(fig_size, fig_size))
        nx.draw_networkx(graph, pos=layout, node_size=nodes_size) # draw graph
        plt.savefig('graph.svg')
        plt.show()
        print("graph plotted in ", (time.time() - start_time)*1000, "ms")


    def print_graph(self):
        """
        Prints formatted graph content
        :return: None
        """

        print("Nodes:")
        for key, value in self.nodes.items():
            print(key)
            for i in self.edges.keys():
                if key in i:
                    print("  ", i, self.edges[i].weight)