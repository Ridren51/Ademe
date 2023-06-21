import pandas as pd
from matplotlib import pyplot as plt

from graph import Graph
import random as rd
import time
import psutil
import datetime
import os
from aco import aco
from genetic import genetic


class Utils:
    def instance_starter(self, instance:object=None, instance_size=-1):
        """
        creates a graph instance if none is provided
        :param instance: instance of Graph class
        :param instance_size: number of nodes in the graph to generate
        :return: instance of Graph class
        """

        if instance is None: #if no instance is provided, generate a random one
            grapher = Graph()
            if instance_size != -1:
                grapher.generate_random_graph(instance_size)
            else:
                grapher.generate_random_graph(rd.randint(10, 100))

        else: #if an instance is provided, use it
            grapher = instance
        return grapher

    def performance_test_multiple_instances(self, func, func_params:dict, iterations:int=1, instance_size:int=-1):
        """
        wrapper for performance test on multiple instances of the same size outputting to a csv file
        :param func: function to test
        :param func_params: default parameters of the function
        :param iterations: number of iterations to run the test
        :param instance_size: number of nodes in the graph
        :return None
        """

        import csv
        import os
        print("Running performance test for ", func.__name__, " with ", iterations, " iterations and ", instance_size, " nodes")

        filename=f'vendor/benchmarks/{func.__name__}/'
        os.makedirs(os.path.dirname(filename), exist_ok=True) #create folder if it doesn't exist

        with open(f"{filename}/{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.csv", mode='w', newline='') as benchfile: #open file
            writer = csv.writer(benchfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) #create csv writer
            writer.writerow(["iteration", "runtime (ms)", "CPU time (ms)", "memory (mb)", "nb_nodes", "nb_edges", "cost", "path"]) #write header

            for iteration in range(iterations):
                grapher=self.instance_starter(instance_size=instance_size) #create graph instance

                start_time = time.time()
                start_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time before running the algorithm
                start_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes

                try:
                    result = func(graph=grapher,**func_params) #run the algorithm
                except Exception:
                    result = ("error", [])

                end_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time after running the algorithm
                end_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes
                end_time = time.time()

                #write results to csv
                writer.writerow([iteration, (end_time - start_time)*1000, (end_cpu_time - start_cpu_time)*1000, end_memory_usage - start_memory_usage, len(grapher.nodes), len(grapher.edges), result[0],result[1]])
        benchfile.close()

    def performance_test(self, func, func_params:dict, iterations:int=1, instance_size:int=-1, instance: object = None):
        """
        wrapper for performance test on a same instance outputting to a csv file
        :param func: function to test
        :param func_params: default parameters of the function
        :param iterations: number of iterations to run the test
        :param instance_size: number of nodes in the graph
        :param instance: adjacency matrix of the graph
        :return None
        """

        import os
        import csv

        print("Running performance test for ", func.__name__, " with ", iterations, " iterations and ",
              instance_size, " nodes")

        grapher = self.instance_starter(instance=instance, instance_size=instance_size)



        filename = f'vendor/benchmarks/{func.__name__}/'
        os.makedirs(os.path.dirname(filename), exist_ok=True) # create folder if it doesn't exist

        with open(f"{filename}/{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.csv", mode='w', newline='') as benchfile: # open file
            writer = csv.writer(benchfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) # create csv writer
            writer.writerow(["iteration", "runtime (ms)", "CPU time (ms)", "memory (mb)", "nb_nodes", "nb_edges", "cost", "path"]) # write header

            for iteration in range(iterations):
                start_time = time.time()
                start_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time before running the algorithm
                start_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes


                result = func(graph=grapher, **func_params) # run the algorithm

                end_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time after running the algorithm
                end_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes
                end_time = time.time()

                # write results to csv
                writer.writerow([iteration, (end_time - start_time)*1000, (end_cpu_time - start_cpu_time)*1000, end_memory_usage - start_memory_usage, len(grapher.nodes), len(grapher.edges), result[0], result[1]])
        benchfile.close()


    def aco_parameters_test(self, parameters:dict, iterations:int=1000, instance_size:int=-1, instance: object = None, outputfile=''):
        """
        method to test the parameters of the aco algorithm and output the results to a csv file
        :param parameters: dictionary of parameters to test with min and max values ex: {"param1": [min, max], "param2": [min, max]}
        :param iterations: number of iterations to run the test
        :param instance_size: number of nodes in the graph
        :param instance: adjacency matrix of the graph
        :return None
        """
        import os
        import csv


        print("Running parameters test for aco with ", iterations, " iterations and ",
              instance_size, " nodes")

        grapher = self.instance_starter(instance=instance, instance_size=instance_size) #create graph instance

        filename = '../vendor/benchmarks/param_test/'
        os.makedirs(os.path.dirname(filename), exist_ok=True) #create folder if it doesn't exist


        if outputfile == '':
            filename += f'aco_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.csv'
        elif outputfile[-4:] != '.csv':
            filename += f'{outputfile}.csv'
        else:
            filename += f'{outputfile}'

        with open(filename, mode='w', newline='') as benchfile: #open file
            writer = csv.writer(benchfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) #create csv writer
            writer.writerow(["iteration", "runtime (ms)", "CPU time (ms)", "memory (mb)", "nb_nodes", "nb_edges", "parameters", "cost", "path"]) #write header

            for iteration in range(iterations):
                start_time = time.time()
                start_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time before running the algorithm
                start_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes

                #create random parameters in the range specified
                func_params = {'alpha': rd.uniform(parameters['alpha'][0],parameters['alpha'][1]), 'beta': rd.uniform(parameters['beta'][0],parameters['beta'][1])}
                # func_params = {'evaporation': rd.uniform(parameters['evaporation'][0],parameters['evaporation'][1]), 'already_visited_penalty': rd.uniform(parameters['already_visited_penalty'][0],parameters['already_visited_penalty'][1])}
                # func_params = {'iterations': rd.randint(parameters['iterations'][0],parameters['iterations'][1])}
                # func_params = {'num_ants': rd.randint(parameters['num_ants'][0],parameters['num_ants'][1])}

                try:
                    result = aco(graph=grapher, start_node=0, **func_params) #run the algorithm
                except Exception:
                    result = ["error", []]

                end_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time after running the algorithm
                end_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes
                end_time = time.time()

                #write results to csv
                writer.writerow([iteration, (end_time - start_time) * 1000, (end_cpu_time - start_cpu_time) * 1000,
                                 end_memory_usage - start_memory_usage, len(grapher.nodes), len(grapher.edges), func_params, result[0],
                                 result[1]])
        benchfile.close()

    def threeD_plot(self, filename:str="", folder:str="", valueA:str="", valueB:str="", shownValueA:str="", shownValueB:str="",title:str=""):
        """
        method to plot the results of the aco parameters test
        :param folder: folder where the csv file is located
        :param filename: name of the csv file
        :return: None
        """

        # plot a 3d graph of the results with x and y being the alpha, beta and z being the cost
        cost = [[], [], []]

        if folder != "":
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                df = pd.read_csv(file_path)

                params = [eval(i) for i in df['parameters'].values]

                value1 = [i[valueA] for i in params]
                value2 = [i[valueB] for i in params]
                # replace strings by 0

                for i, value in enumerate(df['cost'].values):
                    if type(value) == str:
                        if value.isdigit():
                            cost[2].append(float(value))
                            cost[0].append(value1[i])
                            cost[1].append(value2[i])
                    else:
                        cost[2].append(float(value))
                        cost[0].append(value1[i])
                        cost[1].append(value2[i])

        else:
            df = pd.read_csv(filename)
            params = [eval(i) for i in df['parameters'].values]

            value1 = [i[valueA] for i in params]
            value2 = [i[valueB] for i in params]
            # replace strings by 0

            for i, value in enumerate(df['cost'].values):
                if type(value) == str:
                    if value.isdigit():
                        cost[2].append(float(value))
                        cost[0].append(value1[i])
                        cost[1].append(value2[i])
                else:
                    cost[2].append(float(value))
                    cost[0].append(value1[i])
                    cost[1].append(value2[i])


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        ax.figure.set_size_inches(10, 10)
        ax.scatter(cost[0], cost[1], cost[2], c=cost[2], marker='o')
        ax.set_xlabel(shownValueA)
        ax.set_ylabel(shownValueB)
        ax.set_zlabel('Cout')

        ax.stem(cost[0], cost[1], cost[2], linefmt='grey', markerfmt=' ', basefmt=' ')

        plt.show()

    def twoD_plot(self, filename:str="", folder:str="", value:str="", shownValue:str="",title:str=""):
        """
        method to plot the results of the aco parameters test
        :param filename:  name of the csv file
        :param folder:  use a whole folder of csv files
        :param value:  parameter to plot
        :param shownValue:  name of the parameter to show on the graph
        :return:  None
        """

        x = []
        y = []

        if folder != "":
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                df = pd.read_csv(file_path)

                params = [eval(i) for i in df['parameters'].values]

                xs = [i[value] for i in params]
                # replace strings by 0

                for i, v in enumerate(df['cost'].values):
                    if type(v) == str:
                        if v.isdigit():
                            y.append(float(v))
                            x.append(xs[i])
                    else:
                        y.append(v)
                        x.append(xs[i])

        else:
            file_path = os.path.join(folder, filename)
            df = pd.read_csv(file_path)

            params = [eval(i) for i in df['parameters'].values]

            xs = [i[value] for i in params]
            # replace strings by 0

            for i, v in enumerate(df['cost'].values):
                if type(v) == str:
                    if v.isdigit():
                        y.append(float(v))
                        x.append(xs[i])
                    else:
                        y.append(v)
                        x.append(xs[i])

        plt.figure(figsize=(10, 10))
        plt.scatter(x, y)
        plt.title(title)
        plt.xlabel(shownValue)
        plt.ylabel('Cout')

        plt.show()


    def genetic_parameters_test(self, parameters:dict, iterations:int=20, instance_size:int=-1, instance: object = None):
        """
        method to test the parameters of the genetic algorithm and output the results to a csv file
        :param parameters: dictionary of parameters to test with min and max values ex: {"param1": [min, max], "param2": [min, max]}
        :param iterations: number of iterations to run the test
        :param instance_size: number of nodes in the graph
        :param instance: adjacency matrix of the graph
        :return None
        """
        import os
        import csv


        print("Running parameters test for genetic with ", iterations, " iterations and ",
              instance_size, " nodes")

        grapher = self.instance_starter(instance=instance, instance_size=instance_size) #create graph instance

        filename = '../vendor/benchmarks/param_test/'
        os.makedirs(os.path.dirname(filename), exist_ok=True) #create folder if it doesn't exist



        with open(f"{filename}genetic_{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.csv", mode='w', newline='') as benchfile: #open file
            writer = csv.writer(benchfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) #create csv writer
            writer.writerow(["iteration", "runtime (ms)", "CPU time (ms)", "memory (mb)", "nb_nodes", "nb_edges", "parameters", "cost", "path"]) #write header

            for iteration in range(iterations):
                start_time = time.time()
                start_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time before running the algorithm
                start_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes

                #create random parameters in the range specified
                func_params = {'nb_generations': parameters['nb_generations'],'nb_solutions': parameters['nb_solutions'],'nb_kept_solutions':parameters['nb_kept_solutions'], 'mutation_rate': rd.uniform(parameters['mutation_rate'][0],parameters['mutation_rate'][1]), 'cross_over_rate': rd.uniform(parameters['cross_over_rate'][0],parameters['cross_over_rate'][1])}
                print(func_params)

                try:
                    print('try')
                    result = genetic(start_node='0',graph=grapher, **func_params) #run the algorithm
                except Exception as e:
                    print('except', e)
                    result = ["error", []]

                end_cpu_time = psutil.Process().cpu_times().user  # Measure CPU time after running the algorithm
                end_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to megabytes
                end_time = time.time()

                #write results to csv
                writer.writerow([iteration, (end_time - start_time) * 1000, (end_cpu_time - start_cpu_time) * 1000,
                                 end_memory_usage - start_memory_usage, len(grapher.nodes), len(grapher.edges), func_params, result[0],
                                 result[1]])
        benchfile.close()


