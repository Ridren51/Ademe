from graph import Graph
import random as rd
import time
import psutil
import datetime
from aco import aco

class Utils:
    def instance_starter(self, instance:object=None, instance_size=-1):

        if instance is None:
            grapher = Graph()
            if instance_size != -1:
                grapher.generate_random_graph(instance_size)
            else:
                grapher.generate_random_graph(rd.randint(10, 100))

        else:
            grapher = instance
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

    def performance_test(self, func, func_params:dict, iterations:int=1, instance_size:int=-1, instance: object = None): #wrapper for performance test

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


    def aco_parameters_test(self, parameters:dict, iterations:int=1000, instance_size:int=-1, instance: object = None):    #wrapper which tests a function with varying parameters
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
        filename = '../vendor/benchmarks/param_test/'
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
