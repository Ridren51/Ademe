import numpy as np
import random
import copy
import matplotlib.pyplot as plt

X = 1000
Y = 1000
n = 500
price = 1.700

human_cost = 1
ecological_cost = 1
timecost = 1

# EXECUTION SECTION
coordinates = []
distances = []
speeds = []

# COORDINATES GENERATION SECTION
f = open("vendor/Coords/list.txt", "w")
for i in range(1, n+1):
    f.write(str(i) + " " + str(random.randint(1, 1000)) + " " + str(random.randint(1, 1000)) + "\n")
f.close()

# COORDINATES SELECTION SECTION
f = open("vendor/Coords/list.txt", "r")
lines = f.readlines()
f.close()
for line in lines:
    parts = line.split()
    x = int(parts[1])
    y = int(parts[2])
    coordinates.append((x, y))

# DISTANCES GENERATION SECTION
for i in range(len(coordinates)):
    row = []
    for j in range(len(coordinates)):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[j]
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        row.append(distance)
    distances.append(row)

# SPEEDS GENERATION SECTION
for i in range(n):
    row = []
    for j in range(n):
        if i == j:
            distance = 0  # Distances between the same node are 0
        else:
            distance = random.choice([30, 40, 50, 70, 90, 110, 130])  # classic speeds limits
        row.append(distance)
    speeds.append(row)

# CONSUMPTION GENERATION SECTION
consumptions = [[0 for _ in range(len(speeds[i]))] for i in range(len(speeds))]
for i in range(len(speeds)):
    for j in range(len(speeds[i])):
        if speeds[i][j] == 30:
            consumptions[i][j] = 55
        elif speeds[i][j] == 40:
            consumptions[i][j] = 48
        elif speeds[i][j] == 50:
            consumptions[i][j] = 44
        elif speeds[i][j] == 70:
            consumptions[i][j] = 33
        elif speeds[i][j] == 90:
            consumptions[i][j] = 38
        elif speeds[i][j] == 110:
            consumptions[i][j] = 44
        elif speeds[i][j] == 130:
            consumptions[i][j] = 51
        else:
            consumptions[i][j] = 0

consumptions = np.divide(consumptions, 100)

# TIME GENERATION SECTION
d = np.array(distances)  # Distance Matrice
v = np.array(speeds)  # Speeds Matrice
times = np.divide(d, np.nan_to_num(v))  # Time = Distance / Speeds

# COST GENERATION SECTION
cost = np.multiply(times, 9)  # Time in hours

# GASOLINE GENERATION SECTION
gasoline = np.multiply(distances, consumptions)  # Gasoline = Distance * Consumption

# GASOLINE COST GENERATION SECTION
gas_cost = np.multiply(gasoline, price)  # Gasoline Cost = Gasoline * Price

# GLOBAL COST GENERATION SECTION
HC = np.multiply(cost, human_cost)  # Cost = Time * Human Cost
GC = np.multiply(gas_cost, ecological_cost)  # Gasoline Cost = Gasoline * Ecological Cost
T = np.multiply(times, timecost)  # Time = Time * Time Cost

global_cost = np.add(HC, GC, T)  # Global Cost = Cost + Gasoline Cost + Time
global_cost = np.nan_to_num(global_cost, nan=0)
np.savetxt('vendor/Coords/matrix.txt', global_cost, fmt='%.2f')
# PRINT SECTION

# print("Matrice de cout globale:")
# for row in global_cost:
#    print(row)
