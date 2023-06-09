import math
import time
from itertools import combinations, groupby


class Edge:
    def __init__(self, node1: str, node2: str, weight: int):
        self.node1 = node1
        self.node2 = node2
        self.weight = weight


class Node:
    def __init__(self, node_name: str):
        self.nodeName = node_name
        self.degree = 0


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
            self.nodes[node2].degree += 1
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
            if node2 in self.nodes:
                self.nodes[node2].degree -= 1

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
        import random as rd
        start_time = time.time()
        p=.1

        """
        Generates a random undirected graph, similarly to an Erdős-Rényi
        graph, but enforcing that the resulting graph is conneted
        """
        edges = combinations(range(nodes), 2)
        G = nx.Graph()
        G.add_nodes_from(range(nodes))
        if p >= 1:
            G = nx.complete_graph(nodes, create_using=G)
        for _, node_edges in groupby(edges, key=lambda x: x[0]):
            node_edges = list(node_edges)
            G.add_edges_from([rd.choice(node_edges), rd.choice(node_edges)])
            for e in node_edges:
                if rd.random() < p:
                    G.add_edge(*e)

        self.add_nodes_from_list(list(G.nodes))
        # self.node_and_edges_from_adjacency_matrix(nx.adjacency_matrix(G).todense())

        structured_edges = [[str(i[0]), str(i[1]), rd.randint(1, 10)] for i in G.edges]
        self.add_edges_from_list(structured_edges)
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
        startTime = time.time()
        self.adjacency_matrix()

        graph = nx.empty_graph()
        graph.add_nodes_from(self.nodes.keys())

        layout = nx.spring_layout(graph, k=0.1, iterations=50)
        for edge in self.edges.values():
            graph.add_edge(edge.node1, edge.node2)

        if len(self.edges)<10:
            edge_labels = dict([((edge.node1, edge.node2), f'{edge.weight}') for edge in self.edges.values()])
            nx.draw_networkx_edge_labels(graph, pos=layout, edge_labels=edge_labels)

        plt.figure(figsize=(120, 120))
        nx.draw_networkx(graph, pos=layout, node_size=1000)
        plt.savefig('graph.png', dpi=80)
        plt.show()
        print("graph plotted in ", (time.time() - startTime)*1000, "ms")


    # def cycleEulerien(self):
    #     # La matrice est passée par référence, on fait donc une copie de la matrice pour éviter d'écraser ses données.
    #     # Comme il faut aussi copier les listes internes, on fait une _copie profonde_
    #     n = len(matrice)  # Nombre de sommet
    #
    #     cycle = deque()  # cycle est le cycle à construire
    #     stack = deque()  # stack est la pile de sommets à traiter
    #     cur = 0  # cur est le sommet courant. On commence avec le premier sommet de la matrice
    #
    #     # On boucle tant qu'il y a des sommets à traiter dans la pile
    #     # ou que le sommet courant possède au moins 1 voisin non traité
    #     while len(stack) > 0 or degreSommetGrapheMatrice(matrice, cur) != 0:
    #         # Si le sommet courant ne possède aucun voisin, on l'ajoute au cycle
    #         # et on revient au sommet ajouté précédemment dans la pile (backtracking)
    #         # qui devient le nouveau sommet courant
    #         if degreSommetGrapheMatrice(matrice, cur) == 0:
    #             cycle.append(cur)
    #             cur = stack.pop()
    #
    #         # S'il a au moins 1 voisin, on l'ajoute à la pile pour y revenir plus tard (backtracking)
    #         # on retire l'arête qu'il partage avec ce voisin, qui devient le sommet courant
    #         else:
    #             for i in range(n):
    #                 if matrice[cur][i]:
    #                     stack.append(cur)
    #                     matrice[cur][i] = 0
    #                     matrice[i][cur] = 0
    #                     cur = i
    #                     break
    #     return cycle

    def print_graph(self):
        print("Nodes:")
        for key, value in self.nodes.items():
            print(key)
            for i in self.edges.keys():
                if key in i:
                    print("  ", i, self.edges[i].weight)


graphe = Graph()
# graphe.add_nodes_from_list(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
# graphe.add_edges_from_list(
#     [
#         ['A', 'B', 1],
#         ['B', 'C', 1],
#         ['B', 'D', 1],
#         ['C', 'D', 1],
#         ['D', 'C', 1],
#         ['D', 'E', 1],
#         ['E', 'F', 1],
#         ['E', 'G', 1],
#         ['F', 'G', 1],
#         ['G', 'A', 1],
#
#
#     ]
# )
# graphe.node_and_edges_from_adjacency_matrix([[0, 1, 0, 0, 0, 0, 1], [1, 0, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0], [0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 1], [0, 0, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 1, 0]])
graphe.generate_random_graph(250)
print(graphe.is_eulerian_path())
graphe.print_adjency_matrix()
graphe.plot_graph()
print(graphe.print_graph())
