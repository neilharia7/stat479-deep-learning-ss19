# from graphviz import Digraph
#
# # Create a new directed graph
# dot = Digraph()
#
# # Add nodes for the variables
# dot.node('p', 'p')
# dot.node('q', 'q')
# dot.node('r', 'r')
#
# # Add nodes for the intermediate computations
# dot.node('sum1', 'p + q')
# dot.node('sum2', 'q + r')
#
# # Add node for the final computation
# dot.node('u', 'u = (p + q) * (q + r)')
#
# # Add edges to represent the flow of computations
# dot.edge('p', 'sum1')
# dot.edge('q', 'sum1')
# dot.edge('q', 'sum2')
# dot.edge('r', 'sum2')
# dot.edge('sum1', 'u')
# dot.edge('sum2', 'u')
#
# # Render the graph
# dot.render('computational_graph', format='png', cleanup=True)


import matplotlib.pyplot as plt

# Create a new figure
fig, ax = plt.subplots()

# Define node positions
positions = {
    'p': (0, 2),
    'q': (1, 2),
    'r': (2, 2),
    'sum1': (0.5, 1),
    'sum2': (1.5, 1),
    'u': (1, 0)
}

# Draw nodes
for node, (x, y) in positions.items():
    ax.text(x, y, node, fontsize=12, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle'))

# Draw edges
edges = [
    ('p', 'sum1'),
    ('q', 'sum1'),
    ('q', 'sum2'),
    ('r', 'sum2'),
    ('sum1', 'u'),
    ('sum2', 'u')
]

for start, end in edges:
    start_pos = positions[start]
    end_pos = positions[end]
    ax.annotate('', xy=end_pos, xytext=start_pos,
                arrowprops=dict(arrowstyle='->', lw=1.5))

# Set limits and aspect
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.set_aspect('equal')
ax.axis('off')  # Turn off the axis

# Show the plot
plt.show()