from graph import Graph
from utils import Utils
from aco import aco
# utils = Utils()
#
# # utils.performance_test(aco, {'start_node' : 0}, 10, 10)
# # utils.aco_parameters_test(aco, {'start_node' : 0, "iterations":10}, {
# #     'num_ants' : [1, 20],
# #     'alpha' : [1, 20],
# #     'beta' : [1, 20],
# #     'evaporation' : [0.1, 0.9],
# #     'already_visited_penalty' : [0.1, 0.9],
# # }, 10, 10)
#
# grphe = Graph()
# grphe.generate_random_graph(10,p=1)
# # grphe.node_and_edges_from_adjacency_matrix([[0, 13, 32, 0, 13, 0, 0, 48, 0, 30], [13, 0, 22, 0, 21, 0, 65, 21, 0, 0], [32, 22, 0, 0, 53, 0, 14, 0, 40, 16], [0, 0, 0, 0, 38, 34, 0, 0, 9, 88], [13, 21, 53, 38, 0, 5, 0, 39, 37, 59], [0, 0, 0, 34, 5, 0, 0, 21, 0, 26], [0, 65, 14, 0, 0, 0, 0, 0, 24, 44], [48, 21, 0, 0, 39, 21, 0, 0, 66, 0], [0, 0, 40, 9, 37, 0, 24, 66, 0, 60], [30, 0, 16, 88, 59, 26, 44, 0, 60, 0]])
# # # grphe.generate_random_graph(10)
# # # print(grphe.adjacencyMatrix)
# grphe.plot_graph()
# grphe.print_graph()
# # # aco(grphe, 0, num_ants=100, iterations=10)
# utils.aco_parameters_test(parameters={
#                           'num_ants' : [1, 100],
#                           'alpha' : [0, 100],
#                           'beta' : [0, 100],
#                           'evaporation' : [0, 1],
#                           'already_visited_penalty' : [0, 1],
#                           }, iterations=1000, instance=grphe)


graph = Graph()

graph.graph_from_coordinates(["6734;1453",
"2233;10",
"5530;1424",
"401;841",
"3082;1644",
"7608;4458",
"7573;3716",
"7265;1268",
"6898;1885",
"1112;2049",
"5468;2606",
"5989;2873",
"4706;2674",
"4612;2035",
"6347;2683",
"6107;669",
"7611;5184",
"7462;3590",
"7732;4723",
"5900;3561",
"4483;3369",
"6101;1110",
"5199;2182",
"1633;2809",
"4307;2322",
"675;1006",
"7555;4819",
"7541;3981",
"3177;756",
"7352;4506",
"7545;2801",
"3245;3305",
"6426;3173",
"4608;1198",
"23;2216",
"7248;3779",
"7762;4595",
"7392;2244",
"3484;2829",
"6271;2135",
"4985;140",
"1916;1569",
"7280;4899",
"7509;3239",
"10;2676",
"6807;2993",
"5185;3258",
"3023;1942"])

# graph.graph_from_coords_file("../vendor/Coords/list.txt")
graph.graph_from_matrix_file("../vendor/Coords/matrix.txt")

graph.plot_graph()
graph.print_graph()
# # aco(graph, 0, num_ants=100, iterations=10)
# utils = Utils()
# utils.aco_parameters_test(parameters={
#                             'num_ants' : [1, 100],
#                             'alpha' : [0, 10],
#                             'beta' : [0, 10],
#                             'evaporation' : [0, 0.5],
#                             'already_visited_penalty' : [0, 0.5],
#                             }, iterations=1000, instance=graph)


