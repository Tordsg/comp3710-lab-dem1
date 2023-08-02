import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
print("PyTorch Version:", torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rotation_matrix = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32)   
initial_grid = [[0,-1],[0,0]]
initial_array = torch.asarray(initial_grid, dtype=torch.float32)
initial_array.to(device)

#creating the curve of the golden dragon
def curve(x: torch.Tensor):
    #copy and remove the last element
    copy = x.clone()[0:-1]
    rotated = torch.mm(copy,rotation_matrix)
    flipped = torch.flip(rotated, [0])
    concated = torch.cat((x, flipped), 0)
    centered = concated - concated[-1]
    return centered

# Create a figure and axis for the plot
fig, ax = plt.subplots()
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Golden Dragon Fractal')

# Define the number of frames (iterations) for the animation
num_frames = 15

# Update function for the animation
def update(frame):
    ax.clear()
    t = initial_array
    for i in range(frame + 1):
        t = curve(t)
    t = x_ = (t - t.min()) / (t.max() - t.min())
    x_coords = t[:, 0]
    y_coords = t[:, 1]
    ax.plot(x_coords, y_coords, marker='.', linestyle='-', color='b', markersize=1)

# Create the animation
ani = FuncAnimation(fig, update, frames=num_frames, interval=500, repeat=False)

# Show the animation
plt.show()

# t = initial_array

# for i in range(15):
#     t = curve(t)    

# t = x_ = (t - t.min()) / (t.max() - t.min())
# print(t)

# x_coords = t[:, 0]
# y_coords = t[:, 1]

# # Plot the points and connect them with lines
# plt.plot(x_coords, y_coords, marker='.', linestyle='-', color='b', markersize=1)

# # Set axis labels and title
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Golden Dragon Fractal')

# # Show the plot
# plt.show()