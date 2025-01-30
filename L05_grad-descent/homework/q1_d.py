import numpy as np
import matplotlib.pyplot as plt

# Data points
a_values = [0, 0, 0, 1, 1, 1, 2, 2, 2]
b_values = [0, 1, 2, 0, 1, 2, 0, 1, 2]
loss_values = [14, 5, 2, 3, 12, 27, 50, 77, 110]

# Create a grid of a and b values
a_grid = np.array(a_values)
b_grid = np.array(b_values)

# Create the plot
plt.figure(figsize=(10, 8))
plt.scatter(a_grid, b_grid, c=loss_values, cmap='Accent', s=200)

# Add colorbar
plt.colorbar(label='Loss (Sum of Squared Error)')

# Add labels and title
plt.xlabel('a (slope)')
plt.ylabel('b (intercept)')
plt.title('Loss Values on Grid of Parameters')

for i in range(len(a_values)):
    plt.annotate(f'{loss_values[i]}',
                (a_grid[i], b_grid[i]),
                xytext=(5, 5),
                textcoords='offset points')

plt.grid(True)
plt.show()