import numpy as np
import random

n = 10

# EXECUTION SECTION
coordinates = []
distances = []
speeds = []

# COORDINATES GENERATION SECTION
f = open("liste.txt", "w")
for i in range(1, n+1):
    f.write(str(i) + " " + str(random.randint(0, 100)) + " " + str(random.randint(0, 100)) + "\n")
f.close()

# COORDINATES SELECTION SECTION
f = open("liste.txt", "r")
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

# COST GENERATION SECTION

# TIME GENERATION SECTION
d = np.array(distances)  # Distance Matrice
v = np.array(speeds)  # Speeds Matrice
times = np.multiply(d, v)  # Time = Distance / Speeds

# CONSOMPTION GENERATION SECTION

# PRINT SECTION
print("Matrice de distances:")
for row in distances:
    print(row)
print("Matrice de speeds:")
for row in speeds:
    print(row)
print("Matrice de temps:")
for row in times:
    print(row)
print("Matrice de consommation:")
for row in consumptions:
    print(row)
