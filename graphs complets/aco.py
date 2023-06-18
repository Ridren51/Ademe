import time
from graph import Graph
import numpy as np


def aco(graph:Graph, start_node, num_ants:int = 10, alpha:int = 1, beta:int = 2, evaporation:float = 0.5, already_visited_penalty:float = 0.5, iterations:int = 10):
    """
    heuristic aco algorithm for finding the shortest path in a graph
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
    start_time = time.time()

    max_iterations_without_improvement = len(graph.nodes)*2 #max number of iterations without improvement before stopping
    for _ in range(iterations): # Run ant colony optimization for a fixed number of iterations
        print("iteration", _)
        paths = []
        for _ in range(num_ants): # Create ant agents
            # print("ant", _)
            current_city = graph.nodes[start_node] #start from start node
            unvisited_cities = list(graph.nodes.keys()) #list of unvisited cities
            path = []
            edges = []
            cost = 0
            last_city = None

            iterations_without_improvement = 0
            last_size_unvisited_cities=len(unvisited_cities)

            while (unvisited_cities!=[] or current_city.node_name != start_node) :
                # Construct path by iteratively choosing next city until all cities have been visited or if too many iterations without improvement
                neighbor_choice_probabilities = []
                total = 0

                #choose next city
                for neighbor in current_city.neighbors:

                    edge = graph.get_edge(current_city.node_name, neighbor) #get edge between current city and neighbor
                    pheromone = edge.pheromone ** alpha  # Calculate pheromone value
                    distance = 1/edge.weight ** beta if edge.weight != 0 else 0
                    score = pheromone * distance if pheromone != 0 else 1*distance

                    if not unvisited_cities and neighbor == start_node:
                        #if all cities have been visited, give a big bonus to the path that ends in the start node
                        score = score * 1000
                    elif neighbor in path:
                        if neighbor == last_city.node_name:
                            # penalize going back to the last city
                            score = score/1000
                        score = already_visited_penalty * score # penalize already visited cities to avoid loops

                    total += score
                    neighbor_choice_probabilities.append(score)

                probabilities = []
                for p in neighbor_choice_probabilities:
                    if p == 0:
                        probabilities.append(0) # avoid division by 0
                        continue
                    probabilities.append(p / total)  # Calculate probabilities
                if current_city.node_name in unvisited_cities:
                    unvisited_cities.remove(current_city.node_name) #remove current city from unvisited cities
                path.append(current_city.node_name)


                if len(unvisited_cities) < last_size_unvisited_cities:
                    #if we visited a new city, reset the number of iterations without improvement
                    iterations_without_improvement = 0
                else:
                    #if we didn't visit a new city, increment the number of iterations without improvement
                    iterations_without_improvement += 1
                last_size_unvisited_cities = len(unvisited_cities)
                if iterations_without_improvement > max_iterations_without_improvement:
                    #if we didn't visit a new city for too long, stop
                    break

                last_city = current_city
                current_city = graph.nodes[np.random.choice(current_city.neighbors, p=probabilities)]  # Choose next city
                edges.append(graph.get_edge(last_city.node_name, current_city.node_name)) #add edge to path
                cost += graph.get_edge(last_city.node_name, current_city.node_name).weight #add edge weight to cost

            if iterations_without_improvement < max_iterations_without_improvement:
                #avoid adding paths when exceeding the number of iterations without improvement

                #add the return to start node
                path.append(start_node)
                cost += graph.get_edge(last_city.node_name, start_node).weight
                paths.append((cost,path))

                for edge in list(set(edges)): # Update pheromone values on edges
                    edge.pheromone += 1 / cost
                    edge.pheromone *= (1 - evaporation)  # Evaporate pheromone on all edges
        if not paths:
            best_path = ('no reasonable path found', [])
            continue
        best_path = min(paths, key=lambda x: x[0]) # set the new best path
    print("time", (time.time()-start_time)*1000,"ms")
    print("best cost", best_path)
    print('length', len(best_path[1]))

    return best_path # Return best path