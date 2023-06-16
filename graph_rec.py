import matplotlib.pyplot as plt

def read_composite_tours(file_path):
    with open(file_path, 'r') as file:
        tours = []
        tour = []
        for line in file:
            content = line.strip()
            if content == '-----':
                tours.append(tour)
                tour = []
            else:
                tour.append(int(content))
        if tour:
            tours.append(tour)
    return tours


# Path to the text files
file_path = 'vendor/Coords/list.txt'
road_file_path = 'vendor/Coords_rec/road.txt'
composite_file_path = 'vendor/coords_rec/composite_road.txt'

# Lists to store the x and y data
data_x = []
data_y = []

# Read the text file with coordinates
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        parts = line.split()
        x = int(parts[1])
        y = int(parts[2])
        data_x.append(float(x))
        data_y.append(float(y))

# First graph
plt.figure(figsize=(16, 9))
plt.grid(False)
plt.axis('equal')
plt.title('Best Tour Road Graph')
plt.scatter(data_x, data_y, s=150, edgecolors='black', facecolors='white', alpha=0.7)
for i, (x, y) in enumerate(zip(data_x, data_y)):
    plt.text(x, y, f'{i}', ha='center', va='center', color='black', fontsize=7)

# Read the road file and plot the lines
with open(road_file_path, 'r') as road_file:
    road_data = road_file.readlines()
    road_points = [int(point.strip()) for point in road_data]
    for i in range(len(road_points) - 1):
        start_index = road_points[i]
        end_index = road_points[i + 1]
        if 0 <= start_index < len(data_x) and 0 <= end_index < len(data_y):
            plt.plot([data_x[start_index], data_x[end_index]], [data_y[start_index], data_y[end_index]], 'b-')

plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('vendor/img/graph_rec.png')
plt.savefig('vendor/img/graph_rec.svg')
plt.show()

# Second graph
plt.figure(figsize=(16, 9))
plt.grid(False)
plt.axis('equal')
plt.title('Composite Tours Road Graph')
plt.scatter(data_x, data_y, s=150, edgecolors='black', facecolors='white', alpha=0.7)
for i, (x, y) in enumerate(zip(data_x, data_y)):
    plt.text(x, y, f'{i}', ha='center', va='center', color='black', fontsize=7)

# Read the composite file and plot the lines
composite_tours = read_composite_tours(composite_file_path)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for j, road_points in enumerate(composite_tours):
    for i in range(len(road_points) - 1):
        start_index = road_points[i]
        end_index = road_points[i + 1]
        if 0 <= start_index < len(data_x) and 0 <= end_index < len(data_y):
            plt.plot([data_x[start_index], data_x[end_index]], [data_y[start_index], data_y[end_index]], color=colors[j % len(colors)], linestyle='-')

plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('vendor/img/composite_graph_rec.png')
plt.savefig('vendor/img/composite_graph_rec.svg')
plt.show()
