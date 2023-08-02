import torch
import matplotlib.pyplot as plt

# Define the initial grid
initial_grid = [[0, -1], [0, 0]]
initial_array = torch.tensor(initial_grid, dtype=torch.float32)

# Creating the curve of the golden dragon
def curve(x: torch.Tensor):
    # Calculate the new points that connect the existing points
    new_points = x.clone()
    for i in range(len(x) - 1, 0, -1):
        rotation_matrix = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32)
        rotated = torch.mm(new_points[i:] - new_points[i - 1], rotation_matrix)
        new_points[i:] = rotated + new_points[i - 1]
    return new_points

t = initial_array

# Perform 15 iterations and update 't' directly
for i in range(15):
    t = curve(t)

# Separate the x and y coordinates of the points
x_coords = t[:, 0]
y_coords = t[:, 1]

# Plot the points and connect them with lines
plt.plot(x_coords, y_coords, marker='.', linestyle='-', color='b', markersize=1)

# Set axis labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Golden Dragon Fractal')

# Show the plot
plt.show()
