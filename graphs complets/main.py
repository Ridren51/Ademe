from graph import Graph
from utils import Utils

utils = Utils()

# utils.performance_test(aco, {'start_node' : 0}, 10, 10)
# utils.aco_parameters_test(aco, {'start_node' : 0, "iterations":10}, {
#     'num_ants' : [1, 20],
#     'alpha' : [1, 20],
#     'beta' : [1, 20],
#     'evaporation' : [0.1, 0.9],
#     'already_visited_penalty' : [0.1, 0.9],
# }, 10, 10)

grphe = Graph()
grphe.generate_random_graph(10,p=1)
# grphe.node_and_edges_from_adjacency_matrix([[0, 13, 32, 0, 13, 0, 0, 48, 0, 30], [13, 0, 22, 0, 21, 0, 65, 21, 0, 0], [32, 22, 0, 0, 53, 0, 14, 0, 40, 16], [0, 0, 0, 0, 38, 34, 0, 0, 9, 88], [13, 21, 53, 38, 0, 5, 0, 39, 37, 59], [0, 0, 0, 34, 5, 0, 0, 21, 0, 26], [0, 65, 14, 0, 0, 0, 0, 0, 24, 44], [48, 21, 0, 0, 39, 21, 0, 0, 66, 0], [0, 0, 40, 9, 37, 0, 24, 66, 0, 60], [30, 0, 16, 88, 59, 26, 44, 0, 60, 0]])
# # grphe.generate_random_graph(10)
# # print(grphe.adjacencyMatrix)
grphe.plot_graph()
grphe.print_graph()
# # aco(grphe, 0, num_ants=100, iterations=10)
utils.aco_parameters_test(parameters={
                          'num_ants' : [1, 100],
                          'alpha' : [0, 100],
                          'beta' : [0, 100],
                          'evaporation' : [0, 1],
                          'already_visited_penalty' : [0, 1],
                          }, iterations=1000, instance=grphe)