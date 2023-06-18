import random
import time
from collections import defaultdict
from graph import Graph


def genetic(graph, nb_generations, nb_solutions, nb_kept_solutions, mutation_rate, start_node):
    graph.node_and_edges_from_adjacency_matrix([[0, 0, 0, 0, 0, 0, 247, 0, 375, 0], [0, 0, 0, 4, 0, 0, 140, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 323, 457], [0, 4, 0, 0, 0, 0, 0, 287, 0, 0], [0, 0, 0, 0, 0, 0, 334, 0, 0, 116], [0, 0, 0, 0, 0, 0, 552, 0, 0, 485], [247, 140, 0, 0, 334, 552, 0, 0, 0, 0], [0, 0, 0, 287, 0, 0, 0, 0, 373, 0], [375, 0, 323, 0, 0, 0, 0, 373, 0, 0], [0, 0, 457, 0, 116, 485, 0, 0, 0, 0]])
    # graph.generate_random_graph(100)
    graph.plot_graph()

    generation = []
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
        if _ == nb_generations-1:
            print('best found path before crossover and mutation :', generation[0])
        # print('fitness done')

        best_solutions = [generation[0][1]]
        start = time.time()
        best_solutions.extend(
            cross_over(
                generation[i][1], generation[i + 1][1], graph, start_node
            )
            for i in range(nb_kept_solutions - 1)
        )
        print('crossover time :',time.time() - start, 's')
        # print('crossover done')

        start = time.time()
        for i in range(len(best_solutions)):
            best_solutions[i] = mutation(best_solutions[i], mutation_rate, graph)

        print('all mutation time :', time.time() - start, 's')
        solutions = best_solutions
        generation = []

    best_found_path = fitness(graph,solutions, generation)[0]
    print('best found path :',best_found_path)
    print(len(best_found_path[1]))

    print("best path found in", (time.time() - start_time) * 1000, "ms")

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
    def create_indices_dict(parent):
        indices_dict = defaultdict(list)
        for index, node in enumerate(parent):
            indices_dict[node].append(index)
        return indices_dict

    new_path = [start_node]
    nodes_list = list(graph.nodes.keys())
    nodes_list.remove(start_node)

    indices_dict_parent_1 = create_indices_dict(parent_1)
    indices_dict_parent_2 = create_indices_dict(parent_2)

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


def mutation(sol, rate, graph):
    def is_valid_path(path, i1, i2, graph):
        if graph.get_edge(path[i1],path[i1+1]) and graph.get_edge(path[i1], path[i1 - 1]) and graph.get_edge(path[i2], path[i2 + 1]) and graph.get_edge(path[i2], path[i2 - 1]):
            return True
        # return all(graph.get_edge(path[i], path[i + 1]) for i in range(len(path) - 1))

    if random.random() <= rate:
        for _ in range(len(sol)*10):  # Limit to a certain number of tries
            # Choose two node indices at random from the solution
            idx1, idx2 = random.sample(range(1, len(sol)-1), 2)

            if sol[idx1] == sol[idx2]:
                continue

            # Swap the nodes at these indices
            mutated_sol = list(sol)
            mutated_sol[idx1], mutated_sol[idx2] = sol[idx2], sol[idx1]

            # Verify if the mutated solution is still a valid path
            if is_valid_path(mutated_sol, idx1, idx2, graph):
                return mutated_sol
    # If no valid mutation was found, return the original solution
    return sol


genetic_graph = Graph()
# utils = Utils()
# utils.performance_test(genetic,{},10)

nb_gen = 100
nb_sol = 1000
nb_kept_sol = 100
mut_rate = .9
start_n = '0'


genetic(genetic_graph, nb_gen, nb_sol, nb_kept_sol, mut_rate, start_n)
# cProfile.run('genetic(genetic_graph)')