import random
import time
import cProfile
import tsplib95

from main import Graph, Utils


def genetic(graph):
    # graph.node_and_edges_from_adjacency_matrix([[0, 0, 0, 0, 0, 0, 247, 0, 375, 0], [0, 0, 0, 4, 0, 0, 140, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 323, 457], [0, 4, 0, 0, 0, 0, 0, 287, 0, 0], [0, 0, 0, 0, 0, 0, 334, 0, 0, 116], [0, 0, 0, 0, 0, 0, 552, 0, 0, 485], [247, 140, 0, 0, 334, 552, 0, 0, 0, 0], [0, 0, 0, 287, 0, 0, 0, 0, 373, 0], [375, 0, 323, 0, 0, 0, 0, 373, 0, 0], [0, 0, 457, 0, 116, 485, 0, 0, 0, 0]])
    #graph.plot_graph()
    graph.generate_random_graph(100)
    nb_solutions = 20
    nb_kept_solutions = 10
    nb_generations = 50
    generation = []
    start_node = '0'
    solutions = []

    start_time = time.time()

    for _ in range(nb_generations):
        print("generation",_+1)
        if len(solutions) == nb_kept_solutions:
            solutions.extend(
                random_solution(graph, start_node)
                for __ in range(nb_solutions - nb_kept_solutions)
            )
        else:
            solutions.extend(
                random_solution(graph, start_node)
                for __ in range(nb_solutions)
            )

        generation = fitness(graph, solutions, generation)
        # print('fitness done')

        best_solutions = [generation[0][1]]
        # start = time.time()
        best_solutions.extend(
            cross_over(
                generation[i][1], generation[i + 1][1], graph, start_node
            )
            for i in range(nb_kept_solutions - 1)
        )
        # print(time.time() - start, 's')
        # print('crossover done')

        for i in range(len(best_solutions)):
            best_solutions[i] = mutation(best_solutions[i], graph)
        solutions = best_solutions
        generation = []

    best_found_path = fitness(graph,solutions, generation)[0]
    print('best found path :',best_found_path)
    print(len(best_found_path[1]))

    print("graph generated in ", (time.time() - start_time) * 1000, "ms")

    return best_found_path

def fitness(graph, solutions, gen):
    for i in solutions:
        path_cost = sum(
            graph.get_edge(i[j], i[(j + 1)]).weight for j in range(len(i) - 1)
        )
        gen.append((path_cost, i))
    gen= sorted(gen, key=lambda x: x[0])
    return gen

def random_solution(graph, start_node):

    path = []
    nodes_list = list(graph.nodes.keys())
    path.append(start_node)
    nodes_list.pop(nodes_list.index(start_node))
    next_node = random.choice(graph.nodes[start_node].neighbors)
    while nodes_list or path[0] != path[-1]:
        path.append(next_node)
        if next_node in nodes_list:
            nodes_list.pop(nodes_list.index(next_node))

        next_node = random.choice(graph.nodes[next_node].neighbors)

    return path

def cross_over(parent_1,parent_2,graph,start_node):
    new_path = [start_node]
    nodes_list = list(graph.nodes.keys())
    nodes_list.remove(start_node)
    indices_dict_parent_1 = {
        node: [index for index, x in enumerate(parent_1) if x == node]
        for node in set(parent_1)
    }

    indices_dict_parent_2 = {
        node: [index for index, x in enumerate(parent_2) if x == node]
        for node in set(parent_2)
    }

    current_node = start_node
    while nodes_list or new_path[0] != new_path[-1]:
        # indices = [index for index, node in enumerate(parent_1) if node == current_node]
        indices = indices_dict_parent_1.get(current_node, [])
        chosen_index_parent_1 = random.choice(indices)
        next_node_parent_1 = parent_1[chosen_index_parent_1 + 1] if chosen_index_parent_1 < len(parent_1) - 1 else None

        # indices = [index for index, node in enumerate(parent_2) if node == current_node]
        indices = indices_dict_parent_2.get(current_node, [])
        chosen_index_parent_2 = random.choice(indices)
        next_node_parent_2 = parent_2[chosen_index_parent_2 + 1] if chosen_index_parent_2 < len(parent_2) - 1 else None

        # Choose randomly between next_node_parent_1 and next_node_parent_2
        next_node = random.choice([next_node_parent_1, next_node_parent_2])
        if not next_node:
            if valid_next_nodes := [
                node
                for node in graph.nodes[current_node].neighbors
                if node in nodes_list
            ]:
                next_node = random.choice(valid_next_nodes)
            else:
                next_node = random.choice(graph.nodes[current_node].neighbors)

        new_path.append(next_node)
        if next_node in nodes_list:
            nodes_list.remove(next_node)
        current_node = next_node
    return new_path


def mutation(sol, graph):
    for _ in range(100):  # Limit to a certain number of tries
        # Choose two node indices at random from the solution
        idx1, idx2 = random.sample(range(1, len(sol)), 2)

        # Swap the nodes at these indices
        mutated_sol = list(sol)
        mutated_sol[idx1], mutated_sol[idx2] = sol[idx2], sol[idx1]

        # Verify if the mutated solution is still a valid path
        if is_valid_path(mutated_sol, graph):
            # print('mutation success')
            return mutated_sol

    # If no valid mutation was found, return the original solution
    return sol


def is_valid_path(path, graph):
    return all(graph.get_edge(path[i], path[i+1]) for i in range(len(path) - 1))


genetic_graph = Graph()
# utils = Utils()
# utils.performance_test(genetic,{},10)


# genetic(genetic_graph)
cProfile.run('genetic(genetic_graph)')