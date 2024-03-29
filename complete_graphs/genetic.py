import random
import time
from collections import defaultdict
from graph import Graph


def genetic(nb_generations, nb_solutions, nb_kept_solutions, mutation_rate, cross_over_rate, start_node, graph=None):
    """
    Core function of genetic algorithm

    :param nb_generations: Number of generations
    :param nb_solutions: Number of solutions
    :param nb_kept_solutions: Number of solutions retained for next generation
    :param mutation_rate: Chance to apply a mutation operation to a solution
    :param cross_over_rate: Chance to apply a cross-over operation to a solution
    :param start_node: Starting point of the route
    :param graph: Graph to use
    :return: Tuple of the best cost and the list of traveled nodes
    """

    if graph is None:
        graph = Graph()
        graph.generate_random_graph(100)

    # Initialize the list that will store the fitness of each generation.
    generation = []
    solutions = []
    start_time = time.time()

    for _ in range(nb_generations):
        # Generate new solutions for the current generation.
        solutions = generate_solutions(graph, solutions, start_node, nb_solutions, nb_kept_solutions)
        # Evaluate the fitness of each solution in the current generation.
        generation = evaluate_fitness(graph, solutions, generation)

        print(f'Generation: {_ + 1}, Distance: {generation[0][0]} km')
        # Perform the natural selection (selection ,cross-over, mutation) process on the current generation.
        best_solutions = natural_selection(graph, generation, start_node, nb_kept_solutions, cross_over_rate,
                                            mutation_rate)

        solutions = best_solutions
        generation = []

    # Evaluate the fitness of each solution in the last generation.
    best_found_path = evaluate_fitness(graph, solutions, generation)[0]

    # Print result information
    print_best_path_details(best_found_path, start_time)
    return best_found_path

def evaluate_fitness(graph, solutions, gen):
    """
    Computes the total weight of the path in the graph for all solutions in a population.

    :param graph: Graph to use
    :param solutions: Solution population
    :param gen: Empty list ready to receive solutions and their costs
    :return: List of solutions sorted by cost
    """

    for i in solutions:
        path_cost = sum(
            graph.get_edge(i[j], i[(j + 1)]).weight for j in range(len(i) - 1)
        )
        # Append each solution along with its cost to the generation list.
        gen.append((path_cost, i))
    # Sort the generation list by cost, so that the best (lowest cost) solutions are first.
    gen = sorted(gen, key=lambda x: x[0])
    return gen


def generate_solutions(graph, solutions, start_node, nb_solutions, nb_kept_solutions):
    """
    Generates a selected number of random solutions.

    :param graph: Graph to use
    :param solutions: Solution population
    :param start_node: Starting point of the route
    :param nb_solutions: Number of solutions
    :param nb_kept_solutions: Number of solutions retained for next generation
    :return: List of solutions (population)
    """

    if len(solutions) == nb_kept_solutions:
        # Extend the solutions list with new randomly generated solutions, until we reach the required number of solutions for a generation.
        solutions.extend(
            random_solution(graph, start_node)
            for __ in range(nb_solutions - nb_kept_solutions)
        )
    else:
        # Extend the solutions list with new randomly generated solutions, until we reach the required number of solutions for a generation.
        solutions.extend(
            random_solution(graph, start_node)
            for __ in range(nb_solutions)
        )
    return solutions


def random_solution(graph, start_node):
    """
    Creates a random path in the graph starting from a specified node.

    :param graph: Graph to use
    :param start_node: Starting point of the route
    :return: List of traveled nodes
    """

    path = [start_node]
    nodes_list = list(graph.nodes.keys())
    nodes_list.pop(nodes_list.index(start_node))
    next_node = random.choice(graph.nodes[start_node].neighbors)

    # This loop continues until we have visited all nodes and returned back to the start node.
    while nodes_list or path[0] != path[-1]:
        path.append(next_node)
        if next_node in nodes_list:
            nodes_list.pop(nodes_list.index(next_node))

        # Add the next node to the path based on the current node's neighbors
        next_node = random.choice(graph.nodes[next_node].neighbors)

    return path


def natural_selection(graph, generation, start_node, nb_kept_solutions, cross_over_rate, mutation_rate):
    """
    Selects the best solutions and performs crossover and mutation operations.

    :param graph: Graph to use
    :param generation: List of solutions sorted by cost
    :param start_node: Starting point of the route
    :param nb_kept_solutions: Number of solutions retained for next generation
    :param cross_over_rate: Chance to apply a cross-over operation to a solution
    :param mutation_rate: Chance to apply a mutation operation to a solution
    :return: List of traveled nodes of the mutated child
    """

    # Start the new generation with the best solution from the old generation.
    best_solutions = [generation[0][1]]

    #Perform crossover and mutation operations on each solution to generate new solutions for the next generation.
    best_solutions.extend(
        mutate(cross_over(
            generation[i][1],
            generation[i + 1][1],
            graph,
            cross_over_rate,
            start_node,
        ), mutation_rate, graph)
        for i in range(nb_kept_solutions - 1)
    )
    return best_solutions


def cross_over(parent_1, parent_2, graph, cross_over_rate, start_node):
    """
    Performs a crossover operation between two parent solutions to generate a new solution.

    :param parent_1: First parent of the cross-over operation
    :param parent_2: Second parent of the cross-over operation
    :param graph: Graph to use
    :param cross_over_rate: Chance to apply a cross-over operation to a solution
    :param start_node: Starting point of the route
    :return: List of traveled nodes of the child
    """

    # Depending on the crossover rate, this line determines whether the crossover operation should be performed.
    if random.random() > cross_over_rate:
        return parent_1

    # Create an index dictionary for each node
    indices_dict_parent_1 = create_indices_dict(parent_1)
    indices_dict_parent_2 = create_indices_dict(parent_2)

    return create_new_path(
        start_node,
        graph,
        indices_dict_parent_1,
        indices_dict_parent_2,
        parent_1,
        parent_2,
    )


def mutate(sol, mutation_rate, graph):
    """
    Randomly swaps two nodes in a solution to create a mutated solution.

    :param sol: List of traveled nodes after cross-over operation
    :param mutation_rate: Chance to apply a mutation operation to a solution
    :param graph: Graph to use
    :return: List of traveled nodes of the mutated child
    """

    # # Depending on the mutation rate, this line determines whether the mutation operation should be performed.
    if random.random() > mutation_rate:
        return sol

    # Attempt the mutation several times given the low probability of success
    for _ in range(len(sol) * 10):
        # Choose 2 random index in the solution
        idx1, idx2 = random.sample(range(1, len(sol) - 1), 2)

        # if the indexes correspond to the same node, skip this iteration
        if sol[idx1] == sol[idx2]:
            continue

        mutated_sol = list(sol)

        # Invert the 2 nodes corresponding to the indexes
        mutated_sol[idx1], mutated_sol[idx2] = sol[idx2], sol[idx1]

        #Check if the path is valid
        if is_valid_path(mutated_sol, idx1, idx2, graph):
            return mutated_sol

    return sol


def create_indices_dict(parent):
    """
    Creates a dictionary with nodes as keys and their indices in the parent as values.

    :param parent: One of the two chosen parent for the cross-over operation
    :return: Dictionary with nodes as keys and their indices in the parent as values
    """

    indices_dict = defaultdict(list)
    for index, node in enumerate(parent):
        indices_dict[node].append(index)
    return indices_dict


def create_new_path(start_node, graph, indices_dict_parent_1, indices_dict_parent_2, parent_1, parent_2):
    """
    Creates a new path by performing crossover.

    :param start_node: Starting point of the route
    :param graph: Graph to use
    :param indices_dict_parent_1: Dictionary with nodes as keys and their indices in the parent 1 as values
    :param indices_dict_parent_2: Dictionary with nodes as keys and their indices in the parent 2 as values
    :param parent_1: Chosen parent for the cross-over operation
    :param parent_2: Chosen parent for the cross-over operation
    :return: New list of the traveled nodes after cross-over operation
    """

    new_path = [start_node]
    nodes_list = list(graph.nodes.keys())
    # We remove the starting node from the list of nodes we haven't visited yet.
    nodes_list.remove(start_node)

    current_node = start_node

    # This loop continues until we have visited all nodes and returned back to the start node.
    while nodes_list or new_path[0] != new_path[-1]:

        # Selects the next node to be added to the new path.
        next_node = get_next_node(graph, current_node, indices_dict_parent_1, indices_dict_parent_2, nodes_list, parent_1, parent_2)
        new_path.append(next_node)

        # If the next node was in our list of nodes to visit, we remove it.
        if next_node in nodes_list:
            nodes_list.remove(next_node)

        current_node = next_node

    return new_path

def get_next_node(graph, current_node, indices_dict_parent_1, indices_dict_parent_2, nodes_list, parent_1, parent_2):
    """
    Selects the next node to be added to the new path.

    :param graph: Graph to use
    :param current_node: Current node in the new path
    :param indices_dict_parent_1: Dictionary with nodes as keys and their indices in the parent 1 as values
    :param indices_dict_parent_2: Dictionary with nodes as keys and their indices in the parent 2 as values
    :param nodes_list: List of untraveled nodes
    :param parent_1: Chosen parent for the cross-over operation
    :param parent_2: Chosen parent for the cross-over operation
    :return: Next node for the new path
    """

    # We randomly select one of the indices from parent 1 where the current node appears.
    chosen_index_parent_1 = random.choice(indices_dict_parent_1.get(current_node, []))

    # We get the next node in the sequence from parent 1.
    next_node_parent_1 = get_next_node_parent(parent_1, chosen_index_parent_1)


    # We randomly select one of the indices from parent 2 where the current node appears.
    chosen_index_parent_2 = random.choice(indices_dict_parent_2.get(current_node, []))

    # We get the next node in the sequence from parent 2.
    next_node_parent_2 = get_next_node_parent(parent_2, chosen_index_parent_2)

    # We randomly select the next node from the choices given by each parent.
    # If neither parent gives a viable next node, we select a valid node from the graph, preferably one that has never been visited.
    return random.choice(
        [next_node_parent_1, next_node_parent_2]
    ) or get_valid_next_node(graph, current_node, nodes_list)


def get_next_node_parent(parent, chosen_index):
    """
    Gets the next node in the parent solution.

    :param parent: One of the two chosen parent for the cross-over operation
    :param chosen_index: One random index of the current node in the parent
    :return: Next node of one of the parents for the new path
    """

    return parent[chosen_index + 1] if chosen_index < len(parent) - 1 else None


def get_valid_next_node(graph, current_node, nodes_list):
    """
    Selects a valid next node from the graph.

    :param graph: Graph to use
    :param current_node: Current node in the new path
    :param nodes_list: List of untraveled nodes
    :return: Next node for the new path
    """

    # Searches among the neighbours of the current node, if there is a node that has never been visited.
    if valid_next_nodes := [
        node
        for node in graph.nodes[current_node].neighbors
        if node in nodes_list
    ]:
        return random.choice(valid_next_nodes)
    # If not, take a random neighbor for the next node.
    else:
        return random.choice(graph.nodes[current_node].neighbors)


def is_valid_path(path, i1, i2, graph):
    """
    Checks whether a path is valid by verifying that there are edges between each pair of adjacent nodes.

    :param path: Mutated path
    :param i1: Index of an inverted node
    :param i2: Index of another inverted node
    :param graph: Graph to use
    :return: Boolean value indicating whether the path is possible or not
    """

    return graph.get_edge(path[i1], path[i1 + 1]) and graph.get_edge(path[i1], path[i1 - 1]) and graph.get_edge(
        path[i2], path[i2 + 1]) and graph.get_edge(path[i2], path[i2 - 1])


def print_best_path_details(best_found_path, start_time):
    """
    Prints the details of the best found path, including the distance travelled, the number of cities visited, and the computation time.

    :param best_found_path: Tuple of the best cost and the list of traveled nodes
    :param start_time: Algorithm start time
    :return: None
    """

    print(f'Best found path: {best_found_path[1]}')
    print(f'Distance: {best_found_path[0]} km')
    print(f'Cities travelled: {len(best_found_path[1])}')
    print(f"Best path found in {(time.time() - start_time) * 1000} ms")


genetic_graph = Graph()

# genetic_graph.node_and_edges_from_adjacency_matrix(
#     [[0, 0, 0, 0, 0, 0, 247, 0, 375, 0], [0, 0, 0, 4, 0, 0, 140, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 323, 457],
#      [0, 4, 0, 0, 0, 0, 0, 287, 0, 0], [0, 0, 0, 0, 0, 0, 334, 0, 0, 116], [0, 0, 0, 0, 0, 0, 552, 0, 0, 485],
#      [247, 140, 0, 0, 334, 552, 0, 0, 0, 0], [0, 0, 0, 287, 0, 0, 0, 0, 373, 0], [375, 0, 323, 0, 0, 0, 0, 373, 0, 0],
#      [0, 0, 457, 0, 116, 485, 0, 0, 0, 0]])
#genetic_graph.generate_random_graph(nodes=100)
# genetic_graph.graph_from_coordinates(["6734;1453", "2233;10", "5530;1424", "401;841", "3082;1644", "7608;4458",
#                                       "7573;3716", "7265;1268", "6898;1885", "1112;2049", "5468;2606", "5989;2873",
#                                       "4706;2674", "4612;2035", "6347;2683", "6107;669", "7611;5184", "7462;3590",
#                                       "7732;4723", "5900;3561", "4483;3369", "6101;1110", "5199;2182", "1633;2809",
#                                       "4307;2322", "675;1006", "7555;4819", "7541;3981", "3177;756", "7352;4506",
#                                       "7545;2801", "3245;3305", "6426;3173", "4608;1198", "23;2216", "7248;3779",
#                                       "7762;4595", "7392;2244", "3484;2829", "6271;2135", "4985;140", "1916;1569",
#                                       "7280;4899", "7509;3239", "10;2676", "6807;2993", "5185;3258", "3023;1942"])

# genetic_graph.node_and_edges_from_adjacency_matrix([[0, 16, 0, 0, 42, 0, 0, 0, 0, 0, 0, 16, 7, 0, 65, 93, 0, 59, 0, 3, 0, 42, 9, 49, 53, 22, 0, 0, 0, 12, 28, 0, 10, 0, 0, 0, 0, 0, 0, 0, 7, 71, 0, 67, 12, 0, 0, 28, 1, 0, 0, 60, 80, 37, 0, 0, 94, 0, 0, 0, 0, 0, 57, 0, 0, 39, 23, 13, 0, 56, 92, 100, 49, 93, 92, 58, 0, 0, 0, 22, 0, 0, 77, 59, 25, 0, 0, 64, 0, 0, 98, 0, 0, 0, 94, 14, 20, 0, 17, 0], [16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 32, 0, 0, 0, 98, 0, 19, 0, 88, 65, 0, 26, 59, 59, 0, 0, 72, 50, 0, 42, 83, 32, 61, 5, 77, 0, 0, 0, 53, 82, 0, 94, 0, 0, 86, 0, 34, 0, 0, 54, 63, 0, 58, 96, 99, 0, 0, 99, 39, 42, 0, 0, 0, 39, 0, 0, 0, 0, 0, 0, 8, 0, 45, 0, 57, 97, 0, 88, 30, 37, 16, 79, 87, 35, 36, 0, 0, 0, 0, 0, 0, 6, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 0, 0, 0, 80, 88, 78, 0, 93, 94, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 90, 0, 0, 13, 0, 87, 96, 58, 62, 29, 0, 0, 98, 18, 0, 0, 31, 0, 0, 31, 0, 0, 40, 0, 19, 0, 31, 0, 0, 76, 51, 0, 0, 0, 0, 0, 0, 0, 0, 92, 0, 0, 0, 0, 46, 96, 0, 0, 0, 0, 0, 0, 0, 23, 81, 41, 0, 59, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 80, 9, 0, 0, 0, 0, 82, 0, 0, 63, 9, 0, 0, 0, 0, 0, 82, 38, 0, 0, 56, 0, 0, 19, 0, 0, 0, 0, 0, 29, 0, 75, 0, 0, 0, 0, 77, 0, 90, 0, 0, 61, 0, 0, 0, 0, 47, 0, 85, 0, 0, 67, 0, 22, 20, 38, 21, 80, 0, 12, 22, 0, 0, 34, 42, 62, 0, 89, 0, 0, 25, 37, 0, 65, 19, 0, 0, 61, 58, 0, 75, 0, 12, 0, 20, 0, 0, 5, 0], [42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 42, 47, 44, 0, 55, 7, 65, 0, 0, 0, 31, 0, 26, 0, 0, 0, 16, 63, 65, 0, 0, 44, 36, 0, 0, 0, 41, 46, 13, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 44, 0, 0, 0, 0, 55, 0, 0, 58, 95, 0, 46, 81, 0, 0, 71, 0, 0, 93, 0, 99, 44, 0, 21, 0, 84, 5, 59, 52, 0, 25, 0, 0, 38, 0, 0, 0, 24, 0, 81, 91, 51, 0, 0, 0, 97, 0, 87, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 61, 0, 95, 0, 0, 0, 89, 0, 0, 0, 0, 0, 0, 55, 50, 32, 28, 0, 0, 0, 0, 0, 77, 0, 0, 0, 76, 0, 0, 0, 12, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 35, 84, 8, 77, 0, 0, 0, 0, 28, 0, 85, 0, 32, 0, 0, 0, 78, 21, 0, 14, 13, 0, 0, 0, 0, 65, 80, 0, 66, 62, 0, 16, 0, 0, 0, 0, 0, 97, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 0, 0, 0, 0, 0, 0, 21, 0, 0, 1, 6, 93, 0, 0, 96, 0, 76, 0, 0, 14, 0, 29, 9, 0, 0, 0, 98, 91, 34, 11, 76, 49, 0, 73, 0, 0, 0, 21, 15, 0, 0, 0, 9, 14, 23, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 98, 0, 0, 0, 0, 17, 0, 3, 0, 46, 0, 0, 61, 33, 0, 11, 0, 0, 0, 39, 33, 82, 54, 66, 0, 12, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 0, 0, 43, 35, 0, 0, 0, 67, 0, 64, 75, 12, 0, 0, 0, 0, 56, 0, 13, 0, 0, 70, 25, 0, 0, 0, 40, 0, 43, 45, 0, 0, 0, 0, 0, 0, 0, 40, 89, 3, 49, 0, 29, 0, 75, 0, 0, 0, 3, 0, 6, 0, 0, 73, 25, 0, 0, 88, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 77, 0, 0, 0, 7, 0, 0, 33, 0, 0, 64, 56, 0, 0, 31], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 100, 0, 0, 0, 0, 75, 0, 60, 0, 0, 89, 0, 30, 57, 68, 0, 0, 1, 10, 0, 64, 0, 3, 6, 0, 0, 0, 40, 45, 59, 78, 78, 27, 73, 0, 0, 0, 28, 0, 6, 75, 0, 62, 0, 0, 79, 0, 37, 0, 56, 57, 26, 87, 0, 39, 32, 0, 0, 22, 83, 0, 0, 0, 35, 60, 0, 0, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 59, 13, 0, 21, 25, 0, 6, 0, 0, 55], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 67, 0, 0, 0, 0, 0, 60, 84, 0, 0, 0, 9, 82, 0, 91, 0, 0, 0, 0, 0, 0, 0, 39, 0, 0, 0, 98, 59, 0, 0, 75, 38, 22, 21, 0, 0, 88, 90, 0, 64, 0, 0, 0, 82, 94, 0, 52, 0, 0, 36, 25, 0, 0, 90, 1, 0, 75, 30, 0, 9, 44, 40, 0, 79, 0, 0, 0, 74, 0, 0, 8, 0, 76, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 77, 77, 81, 0, 62, 0, 0, 7, 0, 43, 0, 0, 0, 80, 43, 0, 67, 0, 10, 31, 0, 0, 0, 0, 53, 42, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 33, 0, 0, 4, 32, 0, 73, 89, 5, 0, 99, 85, 25, 29, 0, 6, 0, 75, 92, 91, 0, 18, 0, 51, 50, 35, 61, 0, 21, 96, 0, 82, 0, 0, 48, 99, 0, 33, 0, 0, 0, 0, 18, 0, 77, 0, 59, 0, 69, 0, 0, 55, 98, 0, 0, 0, 46, 0, 0, 0, 78, 0], [16, 0, 0, 80, 42, 3, 0, 55, 18, 0, 43, 0, 15, 0, 0, 0, 0, 50, 0, 42, 0, 85, 82, 0, 0, 28, 0, 0, 1, 0, 33, 27, 91, 83, 0, 31, 59, 0, 66, 0, 75, 0, 85, 65, 94, 0, 11, 0, 11, 79, 33, 48, 89, 80, 0, 0, 0, 0, 0, 0, 0, 22, 50, 69, 39, 0, 0, 87, 0, 0, 29, 57, 50, 0, 75, 44, 0, 0, 0, 58, 66, 0, 0, 12, 0, 0, 35, 0, 0, 11, 69, 0, 98, 79, 0, 0, 0, 64, 0, 32], [7, 0, 0, 9, 47, 0, 0, 0, 100, 0, 0, 15, 0, 0, 0, 0, 20, 0, 3, 79, 16, 23, 0, 0, 0, 0, 0, 56, 0, 0, 25, 0, 57, 49, 54, 79, 99, 45, 20, 61, 0, 0, 0, 5, 0, 54, 0, 36, 0, 0, 0, 37, 0, 0, 51, 42, 0, 61, 0, 57, 0, 11, 0, 84, 0, 0, 0, 66, 0, 0, 3, 78, 51, 87, 81, 65, 11, 0, 65, 0, 99, 83, 0, 0, 0, 54, 0, 0, 0, 0, 0, 49, 0, 48, 2, 0, 85, 57, 62, 53], [0, 15, 0, 0, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 23, 91, 0, 73, 0, 43, 0, 0, 86, 0, 45, 39, 11, 0, 0, 63, 12, 0, 0, 41, 0, 50, 74, 48, 0, 36, 0, 97, 58, 0, 97, 0, 21, 0, 90, 0, 0, 20, 0, 0, 6, 16, 11, 0, 0, 0, 0, 0, 0, 0, 27, 25, 0, 81, 0, 35, 44, 11, 0, 86, 90, 0, 0, 92, 15, 0, 35, 0, 6, 6, 0, 24, 69, 0, 46, 38, 18, 36, 0, 0, 0, 57], [65, 0, 80, 0, 0, 0, 0, 43, 0, 67, 0, 0, 0, 3, 0, 0, 22, 0, 95, 25, 93, 83, 0, 0, 18, 34, 0, 84, 58, 0, 72, 0, 0, 0, 99, 0, 26, 0, 0, 25, 0, 0, 0, 0, 0, 0, 18, 12, 33, 0, 97, 68, 0, 22, 47, 4, 61, 94, 52, 66, 53, 20, 0, 7, 10, 19, 64, 0, 32, 0, 0, 67, 60, 10, 53, 0, 0, 0, 0, 0, 75, 0, 0, 12, 35, 91, 0, 56, 0, 0, 47, 89, 0, 49, 0, 0, 61, 87, 37, 0], [93, 0, 88, 0, 55, 0, 0, 35, 0, 0, 80, 0, 0, 0, 0, 0, 0, 16, 0, 0, 53, 0, 34, 77, 81, 75, 0, 15, 83, 0, 0, 0, 12, 0, 7, 0, 30, 0, 0, 65, 4, 0, 0, 0, 0, 0, 6, 19, 0, 25, 0, 75, 0, 37, 14, 0, 0, 65, 0, 0, 88, 0, 35, 0, 14, 0, 0, 0, 14, 41, 25, 100, 67, 0, 0, 90, 95, 0, 0, 0, 0, 48, 50, 0, 1, 0, 0, 46, 0, 44, 0, 99, 0, 64, 30, 67, 16, 0, 0, 50], [0, 32, 78, 0, 7, 61, 0, 0, 0, 0, 43, 0, 20, 0, 22, 0, 0, 0, 53, 37, 0, 58, 0, 70, 64, 83, 90, 73, 67, 84, 70, 0, 0, 13, 0, 33, 0, 0, 0, 3, 0, 58, 95, 99, 87, 0, 0, 74, 0, 0, 0, 0, 99, 59, 0, 6, 0, 0, 0, 0, 49, 0, 57, 84, 17, 88, 95, 0, 58, 55, 9, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 12, 80, 93, 0, 0, 0, 0, 99, 95, 0, 0, 0, 15, 0, 90, 0, 0, 0], [59, 0, 0, 82, 65, 0, 21, 0, 75, 0, 0, 50, 0, 0, 0, 16, 0, 0, 32, 96, 67, 0, 14, 28, 0, 16, 17, 92, 0, 22, 21, 61, 63, 36, 0, 39, 87, 0, 54, 72, 0, 0, 65, 57, 99, 0, 19, 0, 0, 9, 100, 0, 17, 0, 0, 0, 0, 95, 2, 0, 48, 29, 75, 0, 0, 1, 20, 0, 89, 63, 33, 0, 69, 0, 33, 61, 11, 2, 70, 0, 0, 0, 23, 0, 2, 0, 84, 39, 0, 0, 0, 0, 36, 94, 0, 62, 0, 37, 90, 0], [0, 0, 93, 0, 0, 95, 0, 0, 0, 0, 67, 0, 3, 23, 95, 0, 53, 32, 0, 5, 85, 11, 0, 98, 32, 20, 0, 70, 92, 88, 0, 0, 10, 0, 20, 75, 97, 41, 0, 36, 98, 96, 70, 0, 28, 41, 0, 75, 0, 9, 38, 0, 81, 27, 0, 0, 0, 0, 58, 10, 0, 0, 75, 0, 0, 14, 26, 0, 98, 0, 79, 97, 0, 0, 79, 39, 0, 0, 51, 57, 33, 63, 82, 0, 78, 8, 0, 0, 87, 17, 57, 1, 88, 17, 84, 68, 9, 41, 0, 0], [3, 0, 94, 0, 0, 0, 0, 67, 60, 0, 0, 42, 79, 91, 25, 0, 37, 96, 5, 0, 29, 0, 12, 99, 0, 54, 0, 10, 0, 99, 0, 0, 0, 91, 11, 8, 0, 91, 0, 34, 12, 99, 45, 36, 0, 0, 19, 0, 0, 14, 0, 21, 48, 16, 14, 0, 0, 99, 0, 0, 0, 37, 14, 49, 36, 15, 0, 48, 17, 0, 91, 0, 31, 6, 95, 0, 0, 0, 0, 96, 0, 26, 46, 56, 16, 27, 45, 60, 69, 92, 27, 92, 0, 94, 0, 0, 0, 66, 0, 57], [0, 98, 0, 63, 0, 0, 1, 0, 0, 60, 10, 0, 16, 0, 93, 53, 0, 67, 85, 29, 0, 0, 37, 79, 11, 0, 0, 0, 0, 74, 0, 8, 0, 0, 30, 0, 66, 0, 19, 7, 0, 98, 0, 0, 0, 0, 0, 0, 0, 0, 40, 69, 0, 0, 0, 20, 0, 0, 0, 0, 20, 0, 10, 29, 0, 0, 0, 0, 3, 0, 0, 80, 0, 18, 2, 0, 41, 44, 0, 0, 0, 37, 28, 0, 0, 0, 0, 0, 62, 0, 69, 44, 82, 0, 0, 0, 75, 0, 0, 100], [42, 0, 0, 9, 31, 0, 6, 64, 0, 84, 31, 85, 23, 73, 83, 0, 58, 0, 11, 0, 0, 0, 0, 20, 0, 0, 0, 82, 0, 60, 0, 0, 0, 35, 51, 44, 0, 0, 0, 0, 59, 85, 0, 32, 6, 0, 2, 46, 37, 78, 0, 20, 19, 0, 0, 31, 0, 24, 97, 0, 70, 0, 0, 37, 49, 0, 19, 0, 69, 0, 83, 0, 98, 84, 0, 7, 0, 0, 88, 75, 0, 79, 0, 0, 62, 80, 7, 0, 0, 19, 49, 34, 0, 0, 22, 45, 0, 0, 0, 98], [9, 19, 0, 0, 0, 89, 93, 75, 89, 0, 0, 82, 0, 0, 0, 34, 0, 14, 0, 12, 37, 0, 0, 39, 0, 0, 35, 0, 35, 34, 0, 91, 43, 0, 53, 0, 69, 20, 0, 0, 36, 12, 0, 23, 32, 72, 90, 0, 0, 80, 3, 0, 0, 91, 11, 0, 63, 0, 0, 26, 0, 0, 0, 23, 30, 98, 0, 32, 0, 1, 52, 0, 0, 40, 0, 0, 0, 95, 29, 0, 0, 66, 0, 25, 0, 0, 0, 4, 0, 12, 88, 87, 60, 42, 26, 41, 0, 0, 73, 64], [49, 0, 0, 0, 26, 0, 0, 12, 0, 0, 0, 0, 0, 43, 0, 77, 70, 28, 98, 99, 79, 20, 39, 0, 87, 82, 0, 0, 46, 0, 0, 0, 0, 0, 11, 70, 69, 25, 81, 0, 8, 30, 0, 46, 0, 0, 0, 22, 0, 0, 75, 60, 0, 0, 73, 94, 79, 33, 27, 0, 0, 28, 51, 0, 0, 36, 28, 55, 70, 0, 76, 54, 39, 0, 3, 90, 0, 40, 0, 71, 0, 0, 52, 73, 97, 0, 95, 0, 0, 15, 0, 31, 95, 0, 0, 0, 78, 0, 15, 68], [53, 88, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 18, 81, 64, 0, 32, 0, 11, 0, 0, 87, 0, 0, 9, 98, 95, 0, 26, 48, 28, 0, 18, 43, 0, 87, 91, 29, 0, 0, 34, 15, 81, 0, 93, 0, 90, 0, 82, 12, 73, 61, 7, 44, 0, 8, 23, 0, 18, 91, 0, 0, 0, 0, 0, 94, 0, 0, 35, 0, 0, 0, 0, 11, 0, 23, 77, 0, 70, 60, 67, 42, 90, 83, 99, 81, 10, 0, 0, 97, 42, 0, 0, 55, 55, 0, 0, 57], [22, 65, 0, 0, 0, 0, 96, 0, 57, 9, 0, 28, 0, 0, 34, 75, 83, 16, 20, 54, 0, 0, 0, 82, 0, 0, 28, 89, 0, 0, 29, 0, 1, 0, 69, 0, 22, 82, 70, 0, 0, 9, 0, 22, 0, 0, 49, 0, 0, 62, 1, 0, 75, 0, 26, 0, 0, 76, 0, 0, 0, 45, 32, 50, 0, 0, 0, 0, 0, 0, 98, 68, 0, 85, 0, 15, 0, 97, 0, 67, 0, 85, 38, 36, 0, 53, 0, 16, 0, 50, 29, 41, 59, 0, 10, 22, 0, 4, 0, 0], [0, 0, 33, 0, 0, 0, 0, 0, 68, 82, 53, 0, 0, 86, 0, 0, 90, 17, 0, 0, 0, 0, 35, 0, 9, 28, 0, 0, 50, 0, 0, 69, 91, 0, 39, 69, 0, 0, 0, 84, 0, 64, 0, 0, 0, 0, 0, 69, 62, 65, 0, 0, 24, 65, 0, 0, 0, 0, 71, 0, 0, 0, 53, 79, 54, 83, 92, 15, 0, 0, 0, 0, 0, 0, 56, 0, 44, 0, 0, 37, 0, 35, 34, 0, 95, 0, 0, 86, 63, 53, 0, 0, 28, 50, 79, 0, 39, 0, 0, 76], [0, 26, 0, 82, 16, 0, 76, 0, 0, 0, 42, 0, 56, 0, 84, 15, 73, 92, 70, 10, 0, 82, 0, 0, 98, 89, 0, 0, 1, 90, 13, 90, 0, 27, 0, 0, 0, 0, 100, 24, 74, 0, 0, 28, 0, 0, 37, 87, 0, 0, 0, 58, 0, 0, 0, 0, 96, 0, 0, 0, 33, 38, 2, 0, 70, 38, 67, 0, 0, 0, 35, 69, 93, 0, 0, 18, 0, 0, 0, 13, 0, 0, 53, 0, 0, 0, 0, 0, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58], [0, 59, 0, 38, 63, 0, 0, 56, 0, 91, 0, 1, 0, 45, 58, 83, 67, 0, 92, 0, 0, 0, 35, 46, 95, 0, 50, 1, 0, 72, 0, 3, 59, 45, 1, 0, 0, 0, 36, 63, 0, 0, 73, 11, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 73, 0, 17, 4, 0, 0, 47, 97, 0, 20, 0, 50, 0, 19, 0, 0, 0, 1, 0, 37, 83, 0, 0, 18, 9, 0, 15, 13, 0, 44, 80, 48, 0, 78, 30, 56, 0, 78, 0, 0, 0, 0, 91, 75, 94], [12, 59, 0, 0, 65, 55, 0, 0, 1, 0, 0, 0, 0, 39, 0, 0, 84, 22, 88, 99, 74, 60, 34, 0, 0, 0, 0, 90, 72, 0, 79, 50, 20, 65, 89, 0, 90, 51, 0, 83, 0, 0, 99, 0, 62, 14, 0, 15, 0, 56, 39, 35, 0, 60, 0, 23, 0, 0, 72, 0, 0, 27, 81, 22, 96, 48, 37, 77, 9, 0, 41, 57, 0, 0, 38, 0, 0, 39, 10, 0, 0, 0, 21, 83, 0, 50, 55, 0, 0, 52, 11, 57, 62, 0, 0, 0, 94, 0, 57, 73], [28, 0, 0, 0, 0, 50, 14, 13, 10, 0, 0, 33, 25, 11, 72, 0, 70, 21, 0, 0, 0, 0, 0, 0, 26, 29, 0, 13, 0, 79, 0, 0, 0, 0, 0, 17, 25, 9, 0, 0, 0, 67, 84, 16, 72, 0, 0, 0, 0, 0, 23, 45, 74, 47, 93, 55, 0, 0, 79, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 57, 0, 56, 0, 25, 0, 96, 0, 0, 74, 90, 32, 61, 8, 0, 2, 0, 0, 85, 95, 0, 83, 85, 0, 0, 81, 0, 0, 0], [0, 0, 0, 56, 0, 32, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 61, 0, 0, 8, 0, 91, 0, 48, 0, 69, 90, 3, 50, 0, 0, 16, 0, 98, 96, 80, 70, 0, 0, 44, 0, 52, 0, 0, 43, 0, 92, 0, 70, 0, 56, 88, 0, 0, 0, 0, 0, 0, 26, 0, 37, 61, 77, 0, 41, 24, 0, 0, 0, 0, 0, 0, 28, 15, 0, 98, 81, 0, 91, 0, 0, 0, 79, 86, 0, 0, 0, 5, 46, 0, 11, 0, 82, 62, 0, 0, 43, 68, 18], [10, 72, 0, 0, 44, 28, 29, 0, 64, 0, 0, 91, 57, 0, 0, 12, 0, 63, 10, 0, 0, 0, 43, 0, 28, 1, 91, 0, 59, 20, 0, 16, 0, 28, 7, 8, 0, 0, 29, 52, 0, 71, 60, 9, 69, 55, 20, 0, 52, 96, 41, 0, 13, 43, 0, 0, 0, 81, 0, 0, 0, 94, 68, 4, 1, 0, 0, 0, 0, 99, 0, 42, 0, 35, 0, 15, 73, 0, 31, 0, 26, 80, 77, 0, 0, 41, 6, 62, 69, 0, 0, 42, 0, 0, 0, 5, 0, 10, 0, 0], [0, 50, 0, 0, 36, 0, 9, 70, 0, 0, 0, 83, 49, 63, 0, 0, 13, 36, 0, 91, 0, 35, 0, 0, 0, 0, 0, 27, 45, 65, 0, 0, 28, 0, 66, 92, 27, 31, 85, 0, 44, 0, 0, 36, 0, 83, 77, 9, 7, 23, 88, 0, 66, 29, 38, 0, 0, 29, 60, 0, 63, 30, 0, 61, 38, 0, 0, 20, 31, 34, 0, 0, 0, 30, 22, 0, 0, 100, 0, 0, 1, 19, 0, 0, 0, 13, 39, 52, 0, 47, 0, 0, 73, 70, 0, 0, 85, 100, 0, 99], [0, 0, 0, 19, 0, 0, 0, 25, 3, 0, 0, 0, 54, 12, 99, 7, 0, 0, 20, 11, 30, 51, 53, 11, 18, 69, 39, 0, 1, 89, 0, 98, 7, 66, 0, 0, 15, 0, 20, 0, 0, 83, 21, 0, 0, 97, 35, 0, 56, 14, 50, 0, 42, 0, 87, 67, 0, 0, 80, 91, 74, 77, 0, 0, 0, 0, 55, 0, 58, 20, 76, 0, 60, 0, 0, 0, 0, 90, 0, 0, 98, 0, 96, 51, 37, 40, 0, 0, 41, 0, 92, 77, 79, 0, 11, 0, 0, 9, 26, 0], [0, 42, 90, 0, 0, 0, 0, 0, 6, 0, 0, 31, 79, 0, 0, 0, 33, 39, 75, 8, 0, 44, 0, 70, 43, 0, 69, 0, 0, 0, 17, 96, 8, 92, 0, 0, 0, 0, 30, 93, 0, 0, 64, 0, 89, 0, 0, 6, 0, 83, 0, 0, 59, 0, 10, 26, 50, 0, 0, 0, 0, 0, 41, 84, 0, 82, 16, 19, 41, 0, 81, 0, 0, 46, 99, 0, 86, 32, 0, 0, 0, 0, 8, 0, 0, 68, 0, 61, 67, 36, 0, 21, 0, 0, 32, 0, 34, 0, 0, 96], [0, 83, 0, 0, 0, 0, 0, 0, 0, 39, 15, 59, 99, 0, 26, 30, 0, 87, 97, 0, 66, 0, 69, 69, 0, 22, 0, 0, 0, 90, 25, 80, 0, 27, 15, 0, 0, 2, 0, 82, 0, 0, 66, 0, 0, 74, 0, 0, 0, 99, 0, 0, 60, 35, 0, 8, 0, 75, 0, 0, 89, 66, 0, 11, 0, 22, 18, 7, 95, 36, 0, 0, 0, 70, 0, 0, 0, 0, 32, 30, 0, 0, 0, 0, 53, 0, 0, 72, 53, 87, 31, 0, 88, 64, 0, 0, 0, 0, 0, 42], [0, 32, 0, 0, 41, 0, 98, 0, 0, 0, 0, 0, 45, 41, 0, 0, 0, 0, 41, 91, 0, 0, 20, 25, 87, 82, 0, 0, 0, 51, 9, 70, 0, 31, 0, 0, 2, 0, 48, 2, 0, 14, 24, 0, 0, 23, 29, 93, 0, 100, 4, 0, 2, 32, 0, 0, 0, 38, 0, 15, 34, 92, 0, 0, 55, 0, 0, 28, 5, 0, 0, 0, 0, 0, 100, 0, 0, 0, 7, 0, 68, 0, 0, 61, 96, 96, 64, 0, 55, 12, 11, 0, 0, 69, 98, 33, 0, 91, 55, 61], [0, 61, 13, 0, 46, 77, 91, 40, 0, 0, 33, 66, 20, 0, 0, 0, 0, 54, 0, 0, 19, 0, 0, 81, 91, 70, 0, 100, 36, 0, 0, 0, 29, 85, 20, 30, 0, 48, 0, 53, 0, 7, 69, 0, 40, 58, 0, 71, 36, 74, 79, 74, 0, 48, 97, 43, 4, 0, 0, 6, 86, 11, 93, 57, 0, 75, 70, 0, 0, 29, 0, 80, 0, 0, 63, 0, 0, 0, 0, 21, 0, 77, 61, 0, 70, 0, 50, 31, 96, 84, 0, 87, 0, 60, 84, 13, 0, 60, 45, 39], [0, 5, 0, 0, 13, 0, 34, 0, 40, 0, 0, 0, 61, 50, 25, 65, 3, 72, 36, 34, 7, 0, 0, 0, 29, 0, 84, 24, 63, 83, 0, 0, 52, 0, 0, 93, 82, 2, 53, 0, 29, 72, 0, 31, 0, 0, 0, 0, 34, 0, 0, 24, 23, 0, 81, 0, 0, 99, 0, 45, 19, 0, 3, 0, 41, 22, 14, 0, 41, 0, 74, 90, 0, 0, 35, 29, 7, 11, 35, 0, 61, 0, 0, 34, 61, 77, 0, 96, 0, 10, 96, 50, 37, 32, 0, 48, 0, 67, 0, 83], [7, 77, 87, 29, 0, 0, 11, 43, 45, 98, 0, 75, 0, 74, 0, 4, 0, 0, 98, 12, 0, 59, 36, 8, 0, 0, 0, 74, 0, 0, 0, 44, 0, 44, 0, 0, 0, 0, 0, 29, 0, 54, 0, 0, 48, 33, 19, 63, 98, 0, 0, 0, 94, 0, 0, 56, 9, 0, 34, 0, 50, 50, 0, 84, 71, 0, 27, 0, 51, 39, 59, 6, 51, 86, 0, 94, 0, 0, 0, 0, 76, 0, 65, 27, 0, 51, 0, 28, 74, 0, 28, 0, 91, 4, 0, 87, 46, 9, 55, 0], [71, 0, 96, 0, 0, 0, 76, 45, 59, 59, 4, 0, 0, 48, 0, 0, 58, 0, 96, 99, 98, 85, 12, 30, 0, 9, 64, 0, 0, 0, 67, 0, 71, 0, 83, 0, 0, 14, 7, 72, 54, 0, 23, 50, 45, 5, 8, 0, 68, 0, 0, 0, 65, 30, 21, 0, 0, 1, 0, 0, 0, 27, 0, 83, 86, 57, 29, 0, 0, 100, 0, 34, 0, 11, 0, 0, 19, 51, 0, 79, 16, 29, 0, 56, 89, 14, 0, 46, 52, 80, 0, 0, 0, 61, 0, 59, 74, 35, 20, 79], [0, 0, 58, 75, 0, 76, 49, 0, 78, 0, 32, 85, 0, 0, 0, 0, 95, 65, 70, 45, 0, 0, 0, 0, 34, 0, 0, 0, 73, 99, 84, 52, 60, 0, 21, 64, 66, 24, 69, 0, 0, 23, 0, 0, 0, 100, 0, 14, 0, 0, 44, 33, 0, 75, 18, 0, 67, 0, 69, 42, 5, 0, 0, 0, 47, 0, 0, 0, 14, 0, 45, 76, 0, 0, 0, 0, 0, 7, 0, 0, 29, 0, 0, 0, 0, 76, 27, 8, 80, 12, 24, 0, 0, 80, 0, 57, 0, 0, 0, 0], [67, 0, 62, 0, 0, 0, 0, 0, 78, 0, 0, 65, 5, 36, 0, 0, 99, 57, 0, 36, 0, 32, 23, 46, 15, 22, 0, 28, 11, 0, 16, 0, 9, 36, 0, 0, 0, 0, 0, 31, 0, 50, 0, 0, 87, 97, 0, 0, 22, 0, 0, 0, 100, 0, 0, 85, 67, 94, 9, 57, 96, 41, 27, 0, 41, 68, 63, 0, 62, 0, 0, 21, 0, 0, 0, 0, 0, 0, 77, 67, 59, 67, 0, 0, 99, 32, 0, 11, 0, 57, 0, 0, 24, 92, 97, 3, 75, 69, 0, 37], [12, 53, 29, 0, 0, 0, 73, 0, 27, 75, 73, 94, 0, 0, 0, 0, 87, 99, 28, 0, 0, 6, 32, 0, 81, 0, 0, 0, 17, 62, 72, 0, 69, 0, 0, 89, 0, 0, 40, 0, 48, 45, 0, 87, 0, 0, 28, 29, 53, 0, 100, 80, 18, 95, 0, 0, 46, 0, 15, 0, 0, 44, 97, 33, 50, 0, 0, 81, 20, 0, 19, 0, 0, 0, 70, 0, 95, 21, 54, 0, 0, 0, 0, 0, 87, 0, 25, 95, 62, 0, 0, 87, 0, 40, 95, 80, 0, 0, 0, 37], [0, 82, 0, 0, 0, 0, 0, 0, 73, 38, 89, 0, 54, 97, 0, 0, 0, 0, 41, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 14, 0, 43, 55, 83, 97, 0, 74, 23, 58, 0, 33, 5, 100, 97, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 94, 0, 0, 80, 0, 0, 0, 0, 59, 99, 93, 0, 74, 0, 23, 13, 0, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 42, 0, 0, 0, 34, 95, 0, 54, 0, 34, 0, 53, 86, 68, 0, 0, 0, 0, 16], [0, 0, 0, 0, 50, 12, 0, 0, 0, 22, 5, 11, 0, 58, 18, 6, 0, 19, 0, 19, 0, 2, 90, 0, 93, 49, 0, 37, 0, 0, 0, 0, 20, 77, 35, 0, 0, 29, 0, 0, 19, 8, 0, 0, 28, 0, 0, 26, 0, 36, 89, 0, 0, 76, 16, 73, 97, 0, 0, 0, 0, 0, 0, 30, 0, 74, 0, 0, 87, 85, 0, 0, 97, 4, 98, 0, 73, 15, 89, 7, 73, 67, 0, 0, 43, 0, 24, 68, 79, 0, 0, 0, 46, 84, 0, 1, 3, 27, 0, 87], [28, 94, 98, 77, 0, 0, 0, 0, 0, 21, 0, 0, 36, 0, 12, 19, 74, 0, 75, 0, 0, 46, 0, 22, 0, 0, 69, 87, 0, 15, 0, 92, 0, 9, 0, 6, 0, 93, 71, 0, 63, 0, 14, 0, 29, 0, 26, 0, 0, 0, 0, 0, 0, 0, 95, 0, 0, 0, 87, 8, 40, 0, 0, 81, 28, 0, 21, 12, 0, 69, 0, 83, 97, 28, 0, 85, 0, 57, 55, 4, 17, 0, 89, 0, 6, 0, 0, 72, 70, 12, 0, 24, 100, 92, 0, 0, 0, 0, 47, 0], [1, 0, 18, 0, 0, 7, 21, 0, 0, 0, 99, 11, 0, 97, 33, 0, 0, 0, 0, 0, 0, 37, 0, 0, 90, 0, 62, 0, 0, 0, 0, 0, 52, 7, 56, 0, 0, 0, 36, 34, 98, 68, 0, 22, 53, 0, 0, 0, 0, 0, 40, 6, 53, 47, 0, 80, 0, 0, 35, 0, 0, 0, 35, 69, 54, 0, 0, 45, 0, 0, 60, 94, 88, 0, 25, 0, 6, 97, 62, 0, 0, 56, 16, 66, 23, 13, 71, 0, 0, 46, 96, 20, 0, 64, 0, 0, 81, 81, 0, 0], [0, 0, 0, 90, 0, 0, 15, 40, 28, 0, 85, 79, 0, 0, 0, 25, 0, 9, 9, 14, 0, 78, 80, 0, 0, 62, 65, 0, 0, 56, 0, 70, 96, 23, 14, 83, 99, 100, 74, 0, 0, 0, 0, 0, 0, 20, 36, 0, 0, 0, 82, 8, 0, 69, 1, 23, 31, 15, 64, 97, 0, 34, 73, 0, 37, 15, 10, 0, 1, 65, 0, 6, 80, 85, 0, 66, 4, 34, 92, 38, 1, 36, 46, 93, 0, 0, 0, 79, 0, 88, 46, 0, 0, 0, 32, 0, 0, 13, 58, 42], [0, 86, 0, 0, 44, 0, 0, 89, 0, 88, 25, 33, 0, 21, 97, 0, 0, 100, 38, 0, 40, 0, 3, 75, 82, 1, 0, 0, 0, 39, 23, 0, 41, 88, 50, 0, 0, 4, 79, 0, 0, 0, 44, 0, 100, 0, 89, 0, 40, 82, 0, 0, 0, 2, 66, 0, 0, 72, 98, 0, 0, 68, 0, 16, 0, 5, 0, 79, 2, 0, 0, 0, 54, 54, 0, 59, 0, 0, 92, 10, 31, 61, 7, 0, 0, 0, 0, 0, 0, 9, 32, 0, 44, 80, 78, 82, 0, 0, 0, 60], [60, 0, 31, 0, 0, 0, 0, 3, 6, 90, 29, 48, 37, 0, 68, 75, 0, 0, 0, 21, 69, 20, 0, 60, 12, 0, 0, 58, 0, 35, 45, 56, 0, 0, 0, 0, 0, 0, 74, 24, 0, 0, 33, 0, 80, 0, 0, 0, 6, 8, 0, 0, 79, 0, 0, 0, 0, 59, 0, 0, 39, 0, 0, 0, 0, 71, 0, 0, 94, 37, 0, 0, 80, 0, 0, 0, 41, 0, 0, 41, 66, 46, 23, 95, 99, 0, 0, 0, 89, 23, 75, 0, 0, 0, 0, 31, 93, 14, 0, 6], [80, 34, 0, 61, 0, 0, 0, 49, 75, 0, 0, 89, 0, 90, 0, 0, 99, 17, 81, 48, 0, 19, 0, 0, 73, 75, 24, 0, 0, 0, 74, 88, 13, 66, 42, 59, 60, 2, 0, 23, 94, 65, 0, 100, 18, 0, 0, 0, 53, 0, 0, 79, 0, 82, 0, 0, 0, 0, 0, 0, 0, 0, 98, 0, 64, 32, 0, 97, 0, 0, 85, 61, 54, 53, 1, 41, 0, 10, 33, 0, 0, 0, 91, 0, 0, 57, 13, 0, 0, 0, 0, 60, 0, 0, 9, 0, 95, 98, 0, 59], [37, 0, 0, 0, 0, 0, 9, 0, 0, 64, 6, 80, 0, 0, 22, 37, 59, 0, 27, 16, 0, 0, 91, 0, 61, 0, 65, 0, 0, 60, 47, 0, 43, 29, 0, 0, 35, 32, 48, 0, 0, 30, 75, 0, 95, 0, 76, 0, 47, 69, 2, 0, 82, 0, 32, 0, 0, 0, 0, 91, 35, 81, 81, 0, 40, 0, 13, 45, 0, 45, 32, 0, 69, 0, 0, 65, 0, 0, 74, 0, 0, 0, 0, 0, 17, 74, 0, 27, 24, 0, 0, 0, 0, 28, 0, 0, 63, 94, 54, 0], [0, 0, 31, 0, 0, 0, 14, 29, 62, 0, 0, 0, 51, 0, 47, 14, 0, 0, 0, 14, 0, 0, 11, 73, 7, 26, 0, 0, 4, 0, 93, 0, 0, 38, 87, 10, 0, 0, 97, 81, 0, 21, 18, 0, 0, 94, 16, 95, 0, 1, 66, 0, 0, 32, 0, 49, 94, 0, 41, 95, 57, 0, 0, 48, 31, 19, 0, 27, 0, 83, 0, 0, 67, 0, 0, 77, 0, 82, 0, 0, 0, 85, 0, 0, 0, 0, 0, 60, 0, 61, 0, 0, 0, 0, 61, 0, 0, 96, 0, 97], [0, 54, 0, 0, 55, 0, 23, 0, 0, 0, 75, 0, 42, 20, 4, 0, 6, 0, 0, 0, 20, 31, 0, 94, 44, 0, 0, 0, 73, 23, 55, 0, 0, 0, 67, 26, 8, 0, 43, 0, 56, 0, 0, 85, 0, 0, 73, 0, 80, 23, 0, 0, 0, 0, 49, 0, 59, 0, 0, 23, 74, 0, 0, 0, 9, 31, 16, 34, 0, 0, 0, 53, 0, 56, 62, 0, 41, 8, 97, 12, 0, 0, 77, 97, 77, 6, 0, 26, 84, 0, 21, 0, 41, 0, 54, 15, 24, 0, 24, 76], [94, 63, 0, 0, 0, 0, 0, 75, 0, 0, 92, 0, 0, 0, 61, 0, 0, 0, 0, 0, 0, 0, 63, 79, 0, 0, 0, 96, 0, 0, 0, 0, 0, 0, 0, 50, 0, 0, 4, 0, 9, 0, 67, 67, 46, 0, 97, 0, 0, 31, 0, 0, 0, 0, 94, 59, 0, 63, 0, 0, 0, 10, 59, 0, 92, 98, 66, 18, 0, 0, 0, 59, 34, 4, 0, 97, 0, 92, 11, 25, 89, 0, 0, 0, 0, 0, 0, 46, 0, 0, 57, 30, 70, 81, 12, 46, 0, 0, 11, 0], [0, 0, 40, 47, 0, 0, 0, 0, 79, 82, 91, 0, 61, 0, 94, 65, 0, 95, 0, 99, 0, 24, 0, 33, 8, 76, 0, 0, 17, 0, 0, 0, 81, 29, 0, 0, 75, 38, 0, 99, 0, 1, 0, 94, 0, 80, 0, 0, 0, 15, 72, 59, 0, 0, 0, 0, 63, 0, 57, 63, 4, 0, 26, 54, 0, 21, 72, 76, 0, 0, 0, 0, 19, 0, 42, 0, 74, 0, 2, 0, 11, 100, 86, 95, 0, 74, 90, 0, 20, 23, 72, 0, 86, 90, 0, 22, 7, 0, 11, 81], [0, 58, 0, 0, 58, 0, 0, 0, 0, 94, 0, 0, 0, 6, 52, 0, 0, 2, 58, 0, 0, 97, 0, 27, 23, 0, 71, 0, 4, 72, 79, 0, 0, 60, 80, 0, 0, 0, 0, 0, 34, 0, 69, 9, 15, 0, 0, 87, 35, 64, 98, 0, 0, 0, 41, 0, 0, 57, 0, 19, 0, 0, 0, 0, 23, 12, 0, 91, 41, 48, 82, 93, 0, 7, 0, 0, 72, 67, 77, 2, 18, 69, 0, 95, 0, 0, 78, 0, 0, 0, 0, 0, 73, 41, 0, 95, 43, 0, 0, 0], [0, 96, 19, 85, 95, 0, 30, 0, 37, 0, 18, 0, 57, 16, 66, 0, 0, 0, 10, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 91, 0, 0, 15, 6, 45, 0, 0, 42, 57, 0, 0, 0, 8, 0, 97, 0, 0, 0, 91, 95, 23, 0, 63, 19, 0, 0, 35, 16, 0, 36, 0, 40, 0, 0, 87, 38, 0, 0, 0, 0, 0, 0, 8, 0, 7, 100, 16, 61, 100, 0, 0, 0, 41, 0, 55, 0, 20, 0, 61, 26, 0, 0, 0, 0, 0], [0, 99, 0, 0, 0, 61, 0, 3, 0, 52, 0, 0, 0, 11, 53, 88, 49, 48, 0, 0, 20, 70, 0, 0, 18, 0, 0, 33, 0, 0, 0, 0, 0, 63, 74, 0, 89, 34, 86, 19, 50, 0, 5, 96, 0, 0, 0, 40, 0, 0, 0, 39, 0, 35, 57, 74, 0, 4, 0, 0, 0, 13, 0, 16, 5, 45, 0, 0, 0, 0, 97, 76, 0, 34, 0, 0, 0, 0, 0, 89, 76, 48, 76, 0, 0, 0, 70, 32, 0, 27, 0, 0, 42, 92, 18, 0, 57, 0, 92, 22], [0, 0, 31, 0, 46, 35, 0, 0, 56, 0, 51, 22, 11, 0, 20, 0, 0, 29, 0, 37, 0, 0, 0, 28, 91, 45, 0, 38, 47, 27, 0, 37, 94, 30, 77, 0, 66, 92, 11, 0, 50, 27, 0, 41, 44, 0, 0, 0, 0, 34, 68, 0, 0, 81, 0, 0, 10, 0, 0, 35, 13, 0, 0, 33, 0, 0, 87, 52, 74, 97, 18, 0, 0, 76, 90, 0, 11, 27, 0, 14, 61, 25, 87, 84, 0, 0, 0, 36, 2, 53, 29, 0, 79, 0, 0, 90, 83, 13, 53, 0], [57, 0, 0, 67, 81, 84, 0, 6, 57, 0, 50, 50, 0, 0, 0, 35, 57, 75, 75, 14, 10, 0, 0, 51, 0, 32, 53, 2, 97, 81, 0, 61, 68, 0, 0, 41, 0, 0, 93, 3, 0, 0, 0, 27, 97, 59, 0, 0, 35, 73, 0, 0, 98, 81, 0, 0, 59, 26, 0, 16, 0, 0, 0, 0, 85, 0, 84, 82, 91, 5, 2, 47, 92, 82, 29, 0, 76, 16, 44, 0, 0, 0, 0, 88, 43, 0, 0, 56, 0, 80, 0, 0, 11, 0, 0, 0, 0, 39, 52, 42], [0, 99, 0, 0, 0, 8, 0, 0, 26, 36, 35, 69, 84, 0, 7, 0, 84, 0, 0, 49, 29, 37, 23, 0, 0, 50, 79, 0, 0, 22, 0, 77, 4, 61, 0, 84, 11, 0, 57, 0, 84, 83, 0, 0, 33, 99, 30, 81, 69, 0, 16, 0, 0, 0, 48, 0, 0, 54, 0, 0, 16, 33, 0, 0, 68, 21, 91, 54, 40, 73, 18, 82, 0, 16, 0, 0, 0, 100, 0, 0, 0, 92, 0, 67, 82, 0, 0, 0, 0, 0, 0, 0, 0, 0, 71, 26, 0, 0, 27, 0], [0, 39, 76, 22, 0, 77, 0, 0, 87, 25, 61, 39, 0, 0, 10, 14, 17, 0, 0, 36, 0, 49, 30, 0, 0, 0, 54, 70, 20, 96, 0, 0, 1, 38, 0, 0, 0, 55, 0, 41, 71, 86, 47, 41, 50, 93, 0, 28, 54, 37, 0, 0, 64, 40, 31, 9, 92, 0, 23, 36, 5, 0, 85, 68, 0, 0, 84, 85, 35, 0, 0, 55, 70, 12, 0, 91, 32, 88, 81, 0, 48, 9, 0, 83, 49, 0, 73, 68, 0, 0, 65, 56, 3, 38, 0, 41, 0, 0, 0, 0], [39, 42, 51, 20, 71, 0, 0, 73, 0, 0, 0, 0, 0, 0, 19, 0, 88, 1, 14, 15, 0, 0, 98, 36, 0, 0, 83, 38, 0, 48, 23, 41, 0, 0, 0, 82, 22, 0, 75, 22, 0, 57, 0, 68, 0, 0, 74, 0, 0, 15, 5, 71, 32, 0, 19, 31, 98, 21, 12, 0, 45, 0, 0, 21, 0, 0, 0, 13, 23, 59, 48, 0, 89, 83, 0, 85, 77, 0, 0, 0, 0, 54, 54, 0, 52, 73, 74, 0, 1, 56, 0, 0, 0, 52, 0, 89, 0, 0, 0, 50], [23, 0, 0, 38, 0, 0, 0, 25, 39, 0, 21, 0, 0, 0, 64, 0, 95, 20, 26, 0, 0, 19, 0, 28, 0, 0, 92, 67, 50, 37, 0, 24, 0, 0, 55, 16, 18, 0, 70, 14, 27, 29, 0, 63, 0, 74, 0, 21, 0, 10, 0, 0, 0, 13, 0, 16, 66, 72, 0, 40, 0, 87, 84, 91, 84, 0, 0, 32, 0, 0, 99, 75, 0, 89, 0, 30, 0, 58, 0, 0, 0, 0, 23, 64, 29, 0, 0, 0, 0, 35, 43, 0, 0, 69, 86, 0, 0, 0, 0, 0], [13, 0, 0, 21, 0, 0, 0, 0, 32, 90, 96, 87, 66, 0, 0, 0, 0, 0, 0, 48, 0, 0, 32, 55, 94, 0, 15, 0, 0, 77, 0, 0, 0, 20, 0, 19, 7, 28, 0, 0, 0, 0, 0, 0, 81, 0, 0, 12, 45, 0, 79, 0, 97, 45, 27, 34, 18, 76, 91, 0, 0, 52, 82, 54, 85, 13, 32, 0, 64, 0, 0, 35, 0, 0, 0, 49, 0, 80, 17, 4, 0, 26, 0, 0, 0, 0, 15, 0, 0, 0, 0, 63, 0, 37, 0, 0, 0, 0, 50, 60], [0, 0, 0, 80, 93, 0, 0, 0, 0, 1, 0, 0, 0, 27, 32, 14, 58, 89, 98, 17, 3, 69, 0, 70, 0, 0, 0, 0, 19, 9, 0, 0, 0, 31, 58, 41, 95, 5, 0, 41, 51, 0, 14, 62, 20, 23, 87, 0, 0, 1, 2, 94, 0, 0, 0, 0, 0, 0, 41, 0, 0, 74, 91, 40, 35, 23, 0, 64, 0, 29, 18, 20, 0, 0, 7, 0, 72, 0, 0, 0, 0, 100, 14, 90, 54, 0, 7, 0, 4, 0, 1, 0, 0, 37, 27, 86, 16, 0, 18, 93], [56, 39, 0, 0, 0, 28, 0, 88, 0, 0, 82, 0, 0, 25, 0, 41, 55, 63, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 99, 34, 20, 0, 36, 0, 29, 0, 39, 100, 0, 0, 0, 13, 85, 69, 0, 65, 0, 37, 0, 45, 83, 0, 0, 0, 48, 87, 0, 97, 5, 73, 0, 59, 0, 0, 29, 0, 12, 0, 73, 0, 0, 0, 60, 87, 34, 0, 0, 80, 40, 16, 0, 0, 0, 0, 0, 18, 0, 0, 67, 0, 89, 0, 24, 0, 0, 0], [92, 0, 0, 12, 99, 0, 98, 0, 22, 75, 0, 29, 3, 0, 0, 25, 9, 33, 79, 91, 0, 83, 52, 76, 35, 98, 0, 35, 0, 41, 0, 0, 0, 0, 76, 81, 0, 0, 0, 74, 59, 0, 45, 0, 19, 0, 0, 0, 60, 0, 0, 0, 85, 32, 0, 0, 0, 0, 82, 38, 97, 18, 2, 18, 0, 48, 99, 0, 18, 12, 0, 100, 0, 40, 0, 0, 0, 0, 62, 32, 0, 35, 63, 0, 57, 0, 54, 100, 0, 13, 0, 67, 0, 5, 0, 5, 0, 17, 0, 0], [100, 0, 0, 22, 44, 85, 0, 0, 83, 30, 0, 57, 78, 81, 67, 100, 0, 0, 97, 0, 80, 0, 0, 54, 0, 68, 0, 69, 0, 57, 57, 0, 42, 0, 0, 0, 0, 0, 80, 90, 6, 34, 76, 21, 0, 0, 0, 83, 94, 6, 0, 0, 61, 0, 0, 53, 59, 0, 93, 0, 76, 0, 47, 82, 55, 0, 75, 35, 20, 0, 100, 0, 0, 13, 87, 0, 0, 0, 61, 0, 31, 0, 0, 0, 18, 0, 0, 59, 84, 36, 7, 11, 50, 0, 9, 0, 70, 55, 62, 40], [49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 50, 51, 0, 60, 67, 0, 69, 0, 31, 0, 98, 0, 39, 0, 0, 0, 93, 1, 0, 0, 0, 0, 0, 60, 0, 0, 0, 0, 0, 51, 0, 0, 0, 0, 0, 97, 97, 88, 80, 54, 80, 54, 69, 67, 0, 34, 19, 0, 0, 0, 0, 92, 0, 70, 89, 0, 0, 0, 73, 0, 0, 0, 0, 0, 7, 0, 25, 0, 39, 0, 56, 0, 61, 0, 0, 33, 11, 0, 100, 0, 60, 0, 71, 0, 62, 0, 0, 5, 76], [93, 0, 0, 0, 21, 32, 0, 0, 0, 9, 99, 0, 87, 35, 10, 0, 24, 0, 0, 6, 18, 84, 40, 0, 0, 85, 0, 0, 0, 0, 56, 28, 35, 30, 0, 46, 70, 0, 0, 0, 86, 11, 0, 0, 0, 0, 4, 28, 0, 85, 54, 0, 53, 0, 0, 56, 4, 0, 7, 0, 34, 76, 82, 16, 12, 83, 89, 0, 0, 0, 40, 13, 0, 0, 0, 74, 13, 0, 55, 24, 0, 0, 33, 54, 0, 33, 0, 0, 0, 51, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0], [92, 0, 92, 34, 0, 0, 0, 0, 0, 44, 0, 75, 81, 44, 53, 0, 0, 33, 79, 95, 2, 0, 0, 3, 0, 0, 56, 0, 37, 38, 0, 15, 0, 22, 0, 99, 0, 100, 63, 35, 0, 0, 0, 0, 70, 33, 98, 0, 25, 0, 0, 0, 1, 0, 0, 62, 0, 42, 0, 0, 0, 90, 29, 0, 0, 0, 0, 0, 7, 0, 0, 87, 0, 0, 0, 41, 43, 0, 24, 0, 49, 27, 13, 48, 0, 8, 0, 44, 73, 0, 0, 87, 0, 0, 0, 45, 0, 0, 2, 0], [58, 0, 0, 42, 84, 0, 17, 0, 35, 40, 33, 44, 65, 11, 0, 90, 0, 61, 39, 0, 0, 7, 0, 90, 11, 15, 0, 18, 83, 0, 25, 0, 15, 0, 0, 0, 0, 0, 0, 29, 94, 0, 0, 0, 0, 0, 0, 85, 0, 66, 59, 0, 41, 65, 77, 0, 97, 0, 0, 0, 0, 0, 0, 0, 91, 85, 30, 49, 0, 0, 0, 0, 7, 74, 41, 0, 0, 77, 88, 83, 79, 29, 84, 38, 0, 0, 0, 60, 51, 0, 0, 0, 55, 0, 88, 0, 0, 0, 13, 0], [0, 8, 0, 62, 5, 0, 0, 0, 60, 0, 0, 0, 11, 0, 0, 95, 0, 11, 0, 0, 41, 0, 0, 0, 0, 0, 44, 0, 0, 0, 0, 98, 73, 0, 0, 86, 0, 0, 0, 7, 0, 19, 0, 0, 95, 0, 73, 0, 6, 4, 0, 41, 0, 0, 0, 41, 0, 74, 72, 0, 0, 11, 76, 0, 32, 77, 0, 0, 72, 60, 0, 0, 0, 13, 43, 0, 0, 63, 46, 0, 96, 0, 0, 0, 12, 0, 63, 97, 0, 0, 0, 7, 2, 0, 92, 94, 0, 0, 0, 0], [0, 0, 0, 0, 59, 78, 3, 0, 0, 79, 0, 0, 0, 86, 0, 0, 0, 2, 0, 0, 44, 0, 95, 40, 23, 97, 0, 0, 0, 39, 96, 81, 0, 100, 90, 32, 0, 0, 0, 11, 0, 51, 7, 0, 21, 0, 15, 57, 97, 34, 0, 0, 10, 0, 82, 8, 92, 0, 67, 8, 0, 27, 16, 100, 88, 0, 58, 80, 0, 87, 0, 0, 25, 0, 0, 77, 63, 0, 0, 0, 88, 0, 28, 0, 48, 48, 48, 0, 0, 0, 0, 75, 0, 44, 0, 17, 0, 58, 78, 0], [0, 45, 0, 89, 52, 21, 0, 0, 0, 0, 0, 0, 65, 90, 0, 0, 0, 70, 51, 0, 0, 88, 29, 0, 77, 0, 0, 0, 18, 10, 0, 0, 31, 0, 0, 0, 32, 7, 0, 35, 0, 0, 0, 77, 54, 0, 89, 55, 62, 92, 92, 0, 33, 74, 0, 97, 11, 2, 77, 0, 0, 0, 44, 0, 81, 0, 0, 17, 0, 34, 62, 61, 0, 55, 24, 88, 46, 0, 0, 79, 6, 56, 68, 0, 0, 33, 0, 0, 0, 4, 67, 0, 45, 0, 0, 49, 90, 0, 58, 73], [22, 0, 46, 0, 0, 0, 46, 0, 34, 0, 0, 58, 0, 0, 0, 0, 0, 0, 57, 96, 0, 75, 0, 71, 0, 67, 37, 13, 9, 0, 0, 91, 0, 0, 0, 0, 30, 0, 21, 0, 0, 79, 0, 67, 0, 0, 7, 4, 0, 38, 10, 41, 0, 0, 0, 12, 25, 0, 2, 7, 89, 14, 0, 0, 0, 0, 0, 4, 0, 0, 32, 0, 39, 24, 0, 83, 0, 0, 79, 0, 81, 0, 0, 85, 92, 0, 98, 19, 0, 45, 51, 0, 0, 0, 100, 0, 43, 0, 0, 95], [0, 57, 96, 0, 25, 14, 0, 0, 0, 0, 18, 66, 99, 0, 75, 0, 0, 0, 33, 0, 0, 0, 0, 0, 70, 0, 0, 0, 0, 0, 74, 0, 26, 1, 98, 0, 0, 68, 0, 61, 76, 16, 29, 59, 0, 0, 73, 17, 0, 1, 31, 66, 0, 0, 0, 0, 89, 11, 18, 100, 76, 61, 0, 0, 48, 0, 0, 0, 0, 0, 0, 31, 0, 0, 49, 79, 96, 88, 6, 81, 0, 84, 0, 65, 0, 28, 0, 8, 22, 92, 0, 9, 0, 0, 64, 79, 0, 0, 4, 96], [0, 97, 0, 25, 0, 13, 0, 0, 0, 74, 0, 0, 83, 92, 0, 48, 0, 0, 63, 26, 37, 79, 66, 0, 60, 85, 35, 0, 15, 0, 90, 0, 80, 19, 0, 0, 0, 0, 77, 0, 0, 29, 0, 67, 0, 42, 67, 0, 56, 36, 61, 46, 0, 0, 85, 0, 0, 100, 69, 16, 48, 25, 0, 92, 9, 54, 0, 26, 100, 80, 35, 0, 56, 0, 27, 29, 0, 0, 56, 0, 84, 0, 0, 67, 0, 74, 0, 38, 0, 48, 0, 64, 53, 0, 25, 0, 0, 16, 0, 0], [77, 0, 0, 37, 0, 0, 61, 0, 0, 0, 77, 0, 0, 15, 0, 50, 12, 23, 82, 46, 28, 0, 0, 52, 67, 38, 34, 53, 13, 21, 32, 0, 77, 0, 96, 8, 0, 0, 61, 0, 65, 0, 0, 0, 0, 0, 0, 89, 16, 46, 7, 23, 91, 0, 0, 77, 0, 86, 0, 61, 76, 87, 0, 0, 0, 54, 23, 0, 14, 40, 63, 0, 0, 33, 13, 84, 0, 28, 68, 0, 0, 0, 0, 0, 39, 0, 0, 17, 93, 0, 0, 0, 73, 0, 0, 0, 0, 55, 51, 83], [59, 88, 0, 0, 38, 0, 33, 0, 0, 0, 0, 12, 0, 0, 12, 0, 80, 0, 0, 56, 0, 0, 25, 73, 42, 36, 0, 0, 0, 83, 61, 79, 0, 0, 51, 0, 0, 61, 0, 34, 27, 56, 0, 0, 0, 0, 0, 0, 66, 93, 0, 95, 0, 0, 0, 97, 0, 95, 95, 100, 0, 84, 88, 67, 83, 0, 64, 0, 90, 16, 0, 0, 61, 54, 48, 38, 0, 0, 0, 85, 65, 67, 0, 0, 0, 0, 0, 0, 60, 69, 47, 46, 9, 33, 43, 0, 98, 74, 0, 35], [25, 30, 0, 65, 0, 0, 0, 33, 0, 8, 59, 0, 0, 35, 35, 1, 93, 2, 78, 16, 0, 62, 0, 97, 90, 0, 95, 0, 44, 0, 8, 86, 0, 0, 37, 0, 53, 96, 70, 61, 0, 89, 0, 99, 87, 0, 43, 6, 23, 0, 0, 99, 0, 17, 0, 77, 0, 0, 0, 0, 0, 0, 43, 82, 49, 52, 29, 0, 54, 0, 57, 18, 0, 0, 0, 0, 12, 48, 0, 92, 0, 0, 39, 0, 0, 3, 22, 0, 51, 0, 25, 23, 66, 0, 0, 0, 0, 14, 0, 0], [0, 37, 0, 19, 0, 0, 11, 77, 0, 0, 0, 0, 54, 0, 91, 0, 0, 0, 8, 27, 0, 80, 0, 0, 83, 53, 0, 0, 80, 50, 0, 0, 41, 13, 40, 68, 0, 96, 0, 77, 51, 14, 76, 32, 0, 34, 0, 0, 13, 0, 0, 0, 57, 74, 0, 6, 0, 74, 0, 0, 0, 0, 0, 0, 0, 73, 0, 0, 0, 0, 0, 0, 0, 33, 8, 0, 0, 48, 33, 0, 28, 74, 0, 0, 3, 0, 0, 53, 62, 68, 0, 0, 82, 82, 0, 0, 73, 35, 34, 0], [0, 16, 0, 0, 0, 65, 0, 0, 0, 76, 69, 35, 0, 6, 0, 0, 0, 84, 0, 45, 0, 7, 0, 95, 99, 0, 0, 0, 48, 55, 2, 0, 6, 39, 0, 0, 0, 64, 50, 0, 0, 0, 27, 0, 25, 95, 24, 0, 71, 0, 0, 0, 13, 0, 0, 0, 0, 90, 78, 0, 70, 0, 0, 0, 73, 74, 0, 15, 7, 0, 54, 0, 33, 0, 0, 0, 63, 48, 0, 98, 0, 0, 0, 0, 22, 0, 0, 0, 89, 71, 0, 0, 43, 0, 0, 51, 0, 54, 30, 0], [64, 79, 0, 0, 24, 80, 0, 0, 0, 0, 0, 0, 0, 6, 56, 46, 0, 39, 0, 60, 0, 0, 4, 0, 81, 16, 86, 0, 0, 0, 0, 0, 62, 52, 0, 61, 72, 0, 31, 96, 28, 46, 8, 11, 95, 0, 68, 72, 0, 79, 0, 0, 0, 27, 60, 26, 46, 0, 0, 41, 32, 36, 56, 0, 68, 0, 0, 0, 0, 0, 100, 59, 11, 0, 44, 60, 97, 0, 0, 19, 8, 38, 17, 0, 0, 53, 0, 0, 58, 96, 0, 74, 67, 30, 0, 0, 19, 0, 39, 84], [0, 87, 23, 61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 87, 69, 62, 0, 0, 0, 10, 0, 63, 31, 78, 0, 0, 5, 69, 0, 41, 67, 53, 55, 96, 0, 74, 52, 80, 0, 62, 54, 79, 70, 0, 0, 0, 89, 0, 24, 0, 84, 0, 20, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 4, 0, 0, 84, 0, 0, 73, 51, 0, 0, 0, 0, 22, 0, 93, 60, 51, 62, 89, 58, 0, 76, 26, 54, 58, 0, 0, 0, 0, 0, 0, 0], [0, 35, 81, 58, 81, 66, 39, 7, 0, 0, 55, 11, 0, 24, 0, 44, 99, 0, 17, 92, 0, 19, 12, 15, 0, 50, 53, 0, 30, 52, 85, 46, 0, 47, 0, 36, 87, 12, 84, 10, 0, 80, 12, 57, 0, 0, 0, 12, 46, 88, 9, 23, 0, 0, 61, 0, 0, 23, 0, 55, 27, 53, 80, 0, 0, 56, 35, 0, 0, 18, 13, 36, 100, 51, 0, 0, 0, 0, 4, 45, 92, 48, 0, 69, 0, 68, 71, 96, 76, 0, 50, 0, 50, 0, 58, 11, 28, 0, 0, 11], [98, 36, 41, 0, 91, 62, 33, 0, 59, 0, 98, 69, 0, 69, 47, 0, 95, 0, 57, 27, 69, 49, 88, 0, 0, 29, 0, 0, 56, 11, 95, 0, 0, 0, 92, 0, 31, 11, 0, 96, 28, 0, 24, 0, 0, 34, 0, 0, 96, 46, 32, 75, 0, 0, 0, 21, 57, 72, 0, 0, 0, 29, 0, 0, 65, 0, 43, 0, 1, 0, 0, 7, 0, 0, 0, 0, 0, 0, 67, 51, 0, 0, 0, 47, 25, 0, 0, 0, 26, 50, 0, 47, 9, 41, 0, 0, 43, 0, 23, 69], [0, 0, 0, 75, 51, 0, 82, 0, 13, 0, 0, 0, 49, 0, 89, 99, 0, 0, 1, 92, 44, 34, 87, 31, 97, 41, 0, 0, 0, 57, 0, 11, 42, 0, 77, 21, 0, 0, 87, 50, 0, 0, 0, 0, 87, 0, 0, 24, 20, 0, 0, 0, 60, 0, 0, 0, 30, 0, 0, 20, 0, 0, 0, 0, 56, 0, 0, 63, 0, 0, 67, 11, 60, 0, 87, 0, 7, 75, 0, 0, 9, 64, 0, 46, 23, 0, 0, 74, 54, 0, 47, 0, 0, 0, 0, 0, 65, 99, 0, 0], [0, 0, 59, 0, 0, 16, 54, 33, 0, 0, 0, 98, 0, 46, 0, 0, 0, 36, 88, 0, 82, 0, 60, 95, 42, 59, 28, 0, 78, 62, 83, 0, 0, 73, 79, 0, 88, 0, 0, 37, 91, 0, 0, 24, 0, 53, 46, 100, 0, 0, 44, 0, 0, 0, 0, 41, 70, 86, 73, 0, 42, 79, 11, 0, 3, 0, 0, 0, 0, 67, 0, 50, 0, 10, 0, 55, 2, 0, 45, 0, 0, 53, 73, 9, 66, 82, 43, 67, 58, 50, 9, 0, 0, 0, 12, 86, 49, 12, 0, 72], [0, 0, 0, 12, 0, 0, 66, 0, 21, 0, 0, 79, 48, 38, 49, 64, 0, 94, 17, 94, 0, 0, 42, 0, 0, 0, 50, 0, 0, 0, 85, 82, 0, 70, 0, 0, 64, 69, 60, 32, 4, 61, 80, 92, 40, 86, 84, 92, 64, 0, 80, 0, 0, 28, 0, 0, 81, 90, 41, 61, 92, 0, 0, 0, 38, 52, 69, 37, 37, 0, 5, 0, 71, 0, 0, 0, 0, 44, 0, 0, 0, 0, 0, 33, 0, 82, 0, 30, 0, 0, 41, 0, 0, 0, 10, 44, 0, 61, 5, 0], [94, 0, 0, 0, 0, 0, 0, 0, 25, 0, 46, 0, 2, 18, 0, 30, 15, 0, 84, 0, 0, 22, 26, 0, 0, 10, 79, 0, 0, 0, 0, 62, 0, 0, 11, 32, 0, 98, 84, 0, 0, 0, 0, 97, 95, 68, 0, 0, 0, 32, 78, 0, 9, 0, 61, 54, 12, 0, 0, 26, 18, 0, 0, 71, 0, 0, 86, 0, 27, 89, 0, 9, 0, 0, 0, 88, 92, 0, 0, 100, 64, 25, 0, 43, 0, 0, 0, 0, 0, 58, 0, 0, 12, 10, 0, 0, 80, 85, 0, 0], [14, 0, 0, 20, 97, 0, 12, 64, 0, 0, 0, 0, 0, 36, 0, 67, 0, 62, 68, 0, 0, 45, 41, 0, 55, 22, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 33, 13, 48, 87, 59, 57, 3, 80, 0, 1, 0, 0, 0, 82, 31, 0, 0, 0, 15, 46, 22, 95, 0, 0, 90, 0, 26, 41, 89, 0, 0, 86, 0, 5, 0, 62, 0, 45, 0, 94, 17, 49, 0, 79, 0, 0, 0, 0, 0, 51, 0, 0, 11, 0, 0, 86, 44, 0, 0, 96, 0, 100, 18], [20, 0, 0, 0, 0, 0, 0, 56, 6, 0, 0, 0, 85, 0, 61, 16, 90, 0, 9, 0, 75, 0, 0, 78, 55, 0, 39, 0, 0, 94, 81, 0, 0, 85, 0, 34, 0, 0, 0, 0, 46, 74, 0, 75, 0, 0, 3, 0, 81, 0, 0, 93, 95, 63, 0, 24, 0, 7, 43, 0, 57, 83, 0, 0, 0, 0, 0, 0, 16, 24, 0, 70, 0, 0, 0, 0, 0, 0, 90, 43, 0, 0, 0, 98, 0, 73, 0, 19, 0, 28, 43, 65, 49, 0, 80, 96, 0, 59, 0, 13], [0, 6, 0, 0, 87, 0, 0, 0, 0, 0, 0, 64, 57, 0, 87, 0, 0, 37, 41, 66, 0, 0, 0, 0, 0, 4, 0, 0, 91, 0, 0, 43, 10, 100, 9, 0, 0, 91, 60, 67, 9, 35, 0, 69, 0, 0, 27, 0, 81, 13, 0, 14, 98, 94, 96, 0, 0, 0, 0, 0, 0, 13, 39, 0, 0, 0, 0, 0, 0, 0, 17, 55, 0, 0, 0, 0, 0, 58, 0, 0, 0, 16, 55, 74, 14, 35, 54, 0, 0, 0, 0, 99, 12, 61, 85, 0, 59, 0, 0, 1], [17, 0, 0, 5, 0, 97, 0, 0, 0, 0, 78, 0, 62, 0, 37, 0, 0, 90, 0, 0, 0, 0, 73, 15, 0, 0, 0, 0, 75, 57, 0, 68, 0, 0, 26, 0, 0, 55, 45, 0, 55, 20, 0, 0, 0, 0, 0, 47, 0, 58, 0, 0, 0, 54, 0, 24, 11, 11, 0, 0, 92, 53, 52, 27, 0, 0, 0, 50, 18, 0, 0, 62, 5, 0, 2, 13, 0, 78, 58, 0, 4, 0, 51, 0, 0, 34, 30, 39, 0, 0, 23, 0, 0, 5, 0, 100, 0, 0, 0, 40], [0, 0, 0, 0, 0, 0, 0, 31, 55, 0, 0, 32, 53, 57, 0, 50, 0, 0, 0, 57, 100, 98, 64, 68, 57, 0, 76, 58, 94, 73, 0, 18, 0, 99, 0, 96, 42, 61, 39, 83, 0, 79, 0, 37, 37, 16, 87, 0, 0, 42, 60, 6, 59, 0, 97, 76, 0, 81, 0, 0, 22, 0, 42, 0, 0, 50, 0, 60, 93, 0, 0, 40, 76, 0, 0, 0, 0, 0, 73, 95, 96, 0, 83, 35, 0, 0, 0, 84, 0, 11, 69, 0, 72, 0, 0, 18, 13, 1, 40, 0]]
# )
#
# nb_gen = 10000
# nb_sol = 20
# nb_kept_sol = 10
# cross_rate = .16
# mut_rate = .79
# start_n = '0'
#
# genetic(nb_gen, nb_sol, nb_kept_sol, mut_rate, cross_rate, start_n, genetic_graph)
