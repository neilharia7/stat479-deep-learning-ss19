# import numpy as np
# import matplotlib.pyplot as plt
#
# # Generate some linearly separable data
# np.random.seed(0)
# num_points = 100
# X = np.random.randn(num_points, 2)
# y = np.array([1 if x[0] + x[1] > 0 else -1 for x in X])
#
# # Plot the data points
# plt.figure(figsize=(10, 5))
# plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='b', label='Class 1')
# plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='r', label='Class -1')
#
# # Perceptron weights without bias
# w_no_bias = np.array([1, -1])
# # Perceptron weights with bias
# w_with_bias = np.array([1, -1, 0.5])  # Adding a bias term
#
# # Function to plot decision boundary
# def plot_decision_boundary(w, bias=False):
#     if bias:
#         # Decision boundary with bias
#         x_values = np.array([-3, 3])
#         y_values = -(w[0] * x_values + w[2]) / w[1]
#         plt.plot(x_values, y_values, 'g--', label='With Bias')
#     else:
#         # Decision boundary without bias
#         x_values = np.array([-3, 3])
#         y_values = -(w[0] * x_values) / w[1]
#         plt.plot(x_values, y_values, 'k-', label='Without Bias')
#
# # Plot decision boundaries
# plot_decision_boundary(w_no_bias, bias=False)
# plot_decision_boundary(w_with_bias, bias=True)
#
# # Add labels and legend
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Perceptron Decision Boundary with and without Bias')
# plt.legend()
# plt.xlim(-3, 3)
# plt.ylim(-3, 3)
# plt.grid(True)
# plt.show()
