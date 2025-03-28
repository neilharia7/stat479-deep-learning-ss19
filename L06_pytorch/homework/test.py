# # import torch
# # import pandas as pd
# #
# # # Load dataset
# # df = pd.read_csv("linreg-data.csv")
# #
# # # Extract features and target variable
# # X = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float32)
# # y = torch.tensor(df['y'].values + 144, dtype=torch.float32).view(-1, 1)
# #
# # # Initialize parameters
# # torch.manual_seed(42)
# # W = torch.rand((2, 1), dtype=torch.float32, requires_grad=False)
# # b = torch.rand(1, dtype=torch.float32, requires_grad=False)
# #
# # # Training settings
# # learning_rate = 0.01
# # epochs = 100
# # n_samples = len(y)
# #
# # # Training loop - Manual gradients
# # for epoch in range(epochs):
# #     y_pred = X @ W + b
# #     loss = ((y_pred - y) ** 2).mean()
# #
# #     # Compute gradients manually
# #     grad_W = (2 / n_samples) * X.T @ (y_pred - y)
# #     grad_b = (2 / n_samples) * (y_pred - y).sum()
# #
# #     # Update parameters
# #     W -= learning_rate * grad_W
# #     b -= learning_rate * grad_b
# #
# #     if epoch % 10 == 0:
# #         print(f"Epoch {epoch}: Loss = {loss.item()}")
# #
# # print("Final parameters:", W, b)
#
#
# import torch
# import pandas as pd
#
#
# class LinearRegressionModel:
# 	def __init__(self, input_size, learning_rate=0.01):
# 		"""
# 		Initialize model parameters and learning rate.
# 		"""
# 		torch.manual_seed(42)
# 		self.W = torch.zeros((input_size, 1), dtype=torch.float32)
# 		self.b = torch.zeros(1, dtype=torch.float32)
# 		self.learning_rate = learning_rate
#
# 	def forward(self, X):
# 		"""
# 		Compute the forward pass (prediction).
# 		"""
# 		return X @ self.W + self.b
#
# 	def compute_loss(self, y_pred, y_true):
# 		"""
# 		Compute Mean Squared Error loss.
# 		"""
# 		return ((y_pred - y_true) ** 2).mean()
#
# 	def backward(self):
# 		"""
# 		Perform backpropagation to compute gradients.
# 		"""
# 		self.loss.backward()
#
# 	def update_parameters(self):
# 		"""
# 		Update model parameters using gradient descent.
# 		"""
# 		# with torch.no_grad():
# 		self.W -= self.learning_rate * self.W.grad
# 		self.b -= self.learning_rate * self.b.grad
#
# 		# # Zero the gradients after update
# 		# self.W.grad.zero_()
# 		# self.b.grad.zero_()
#
# 	def train(self, X, y, epochs=100):
# 		"""
# 		Train the model using gradient descent.
# 		"""
# 		for epoch in range(epochs):
# 			# Forward pass
# 			y_pred = self.forward(X)
#
# 			# Compute loss
# 			self.loss = self.compute_loss(y_pred, y)
#
# 			# Backward pass
# 			self.backward()
#
# 			# Update parameters
# 			self.update_parameters()
#
# 			# Print progress
# 			if epoch % 10 == 0:
# 				print(f"Epoch {epoch}: Loss = {self.loss.item()}")
#
# 		print("Final Parameters:", self.W, self.b)
#
#
# # Load dataset
# df = pd.read_csv("linreg-data.csv")
#
# # Prepare data
# X = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float32)
# y = torch.tensor(df['y'].values + 144, dtype=torch.float32).view(-1, 1)
#
# # Initialize and train model
# model = LinearRegressionModel(input_size=2, learning_rate=0.01)
# model.train(X, y, epochs=100)

#
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# class ManualLinearRegression:
# 	def __init__(self, number_of_features: int, seed: int = 42):
# 		"""
# 		Initialize the linear regression model.
#
# 		:param number_of_features: Number of features in the input.
# 		:param seed: Random seed to ensure reproducibility.
# 		"""
# 		self.number_of_features = number_of_features
# 		self.weights = torch.zeros(number_of_features, 1, dtype=torch.float)
# 		self.bias = torch.zeros(1, dtype=torch.float)
# 		torch.manual_seed(seed)
#
# 	def forward(self, x: torch.Tensor) -> torch.Tensor:
# 		"""
# 		Perform the forward pass of the model.
#
# 		:param x: Input tensor of shape (samples, features).
# 		:return: Predicted values as a 1D tensor.
# 		"""
# 		# Compute weighted sum: x * weights + bias
# 		weighted_sum = torch.add(torch.mm(x, self.weights), self.bias)
# 		# For debugging: print intermediate weighted sums
# 		# print("weighted_sum", weighted_sum)
# 		return weighted_sum.view(-1)
#
# 	def backward(self, x: torch.Tensor, yhat: torch.Tensor, y: torch.Tensor):
# 		"""
# 		Compute gradients for the weights and bias using manual derivation for mean squared error.
#
# 		:param x: Input tensor of shape (samples, features).
# 		:param yhat: Predicted values tensor.
# 		:param y: Actual target values tensor.
# 		:return: Gradients for weights and bias.
# 		"""
# 		# Compute derivative of loss w.r.t predictions, i.e., 2 * (yhat - y)
# 		gradient_loss_yhat = 2 * (yhat - y)
# 		# Gradient with respect to weights: x.T * gradient_loss_yhat averaged over samples
# 		gradient_loss_weights = torch.mm(x.T, gradient_loss_yhat.view(-1, 1)) / y.shape[0]
# 		# Gradient with respect to bias: average of gradient_loss_yhat over samples
# 		gradient_loss_bias = torch.sum(gradient_loss_yhat) / y.shape[0]
# 		# Return negative gradients for gradient descent update
# 		return -gradient_loss_weights, -gradient_loss_bias
#
# 	def loss_function(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
# 		"""
# 		Compute mean squared error loss.
#
# 		:param yhat: Predicted values.
# 		:param y: True values.
# 		:return: Scalar tensor representing the MSE loss.
# 		"""
# 		return torch.mean((yhat - y) ** 2)
#
# 	def train(self, x: torch.Tensor, y: torch.Tensor, epochs: int, learning_rate: float = 0.01):
# 		"""
# 		Train the model using manual gradient descent.
#
# 		:param x: Input tensor of shape (samples, features).
# 		:param y: Target values tensor.
# 		:param epochs: Number of training epochs.
# 		:param learning_rate: Step size for updates.
# 		:return: List of loss values for each epoch.
# 		"""
# 		cost = []
# 		for epoch in range(epochs):
# 			# Forward pass
# 			yhat = self.forward(x)
# 			# Backward pass: compute gradients
# 			negative_gradient_weights, negative_gradient_bias = self.backward(x, yhat, y)
# 			# Update parameters
# 			self.weights += learning_rate * negative_gradient_weights
# 			self.bias += learning_rate * negative_gradient_bias
# 			# Compute new predictions and current loss
# 			yhat = self.forward(x)
# 			current_loss = self.loss_function(yhat, y)
# 			print(f"Epoch: {epoch + 1} | Mean Squared Error: {current_loss:.4f}")
# 			cost.append(current_loss.item())
# 		return cost
#
#
# def generate_synthetic_data(num_samples: int, noise_std: float = 0.5):
# 	"""
# 	Generate synthetic linear data for testing the regression model.
#
# 	:param num_samples: Number of data samples to generate.
# 	:param noise_std: Standard deviation of Gaussian noise added to the output.
# 	:return: Tuple (x, y) of tensors.
# 	"""
# 	# For one-dimensional input feature
# 	x = torch.linspace(0, 10, num_samples).view(-1, 1)
# 	# True relationship: y = 2.5 * x + 1.0 with added noise
# 	true_weights = 2.5
# 	true_bias = 1.0
# 	y = true_weights * x + true_bias + torch.randn(num_samples, 1) * noise_std
# 	# Convert y to 1D tensor for compatibility
# 	return x, y.view(-1)
#
#
# def main():
# 	# Generate synthetic data
# 	x, y = generate_synthetic_data(100)
#
# 	# Initialize the regression model with one input feature
# 	model = ManualLinearRegression(number_of_features=1, seed=42)
#
# 	# Train the model
# 	epochs = 50
# 	learning_rate = 0.01
# 	loss_history = model.train(x, y, epochs=epochs, learning_rate=learning_rate)
#
# 	# Plot loss history
# 	plt.figure(figsize=(8, 4))
# 	plt.plot(loss_history, label="Training Loss")
# 	plt.xlabel("Epoch")
# 	plt.ylabel("Mean Squared Error")
# 	plt.title("Training Loss Over Epochs")
# 	plt.legend()
# 	plt.show()
#
# 	# Visualize the regression line compared to the data
# 	plt.figure(figsize=(8, 4))
# 	plt.scatter(x.numpy(), y.numpy(), label="Data", color="blue")
# 	predictions = model.forward(x).detach().numpy()
# 	plt.plot(x.numpy(), predictions, label="Regression Line", color="red")
# 	plt.xlabel("Input Feature")
# 	plt.ylabel("Target")
# 	plt.title("Manual Linear Regression Fit")
# 	plt.legend()
# 	plt.show()
#
#
# if __name__ == "__main__":
# 	main()


import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ManualLinearRegression:
	def __init__(self, number_of_features: int, seed: int = 42):
		"""
		Initialize the linear regression model.

		:param number_of_features: Number of features in the input.
		:param seed: Random seed to ensure reproducibility.
		"""
		self.number_of_features = number_of_features
		self.weights = torch.zeros(number_of_features, 1, dtype=torch.float)
		self.bias = torch.zeros(1, dtype=torch.float)
		torch.manual_seed(seed)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Perform the forward pass of the model.

		:param x: Input tensor of shape (samples, features).
		:return: Predicted values as a 1D tensor.
		"""
		# Compute weighted sum: x * weights + bias
		weighted_sum = torch.add(torch.mm(x, self.weights), self.bias)
		return weighted_sum.view(-1)

	def backward(self, x: torch.Tensor, yhat: torch.Tensor, y: torch.Tensor):
		"""
		Compute gradients for the weights and bias using manual derivation for mean squared error.

		:param x: Input tensor of shape (samples, features).
		:param yhat: Predicted values tensor.
		:param y: Actual target values tensor.
		:return: Gradients for weights and bias.
		"""
		# Compute derivative of loss w.r.t predictions: 2 * (yhat - y)
		gradient_loss_yhat = 2 * (yhat - y)
		# Gradient with respect to weights: x.T * gradient_loss_yhat averaged over samples
		gradient_loss_weights = torch.mm(x.T, gradient_loss_yhat.view(-1, 1)) / y.shape[0]
		# Gradient with respect to bias: average of gradient_loss_yhat over samples
		gradient_loss_bias = torch.sum(gradient_loss_yhat) / y.shape[0]
		# Return negative gradients for gradient descent update
		return -gradient_loss_weights, -gradient_loss_bias

	def loss_function(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		"""
		Compute mean squared error loss.

		:param yhat: Predicted values.
		:param y: True values.
		:return: Scalar tensor representing the MSE loss.
		"""
		return torch.mean((yhat - y) ** 2)

	def train(self, x: torch.Tensor, y: torch.Tensor, epochs: int, learning_rate: float = 0.01):
		"""
		Train the model using manual gradient descent.

		:param x: Input tensor of shape (samples, features).
		:param y: Target values tensor.
		:param epochs: Number of training epochs.
		:param learning_rate: Step size for updates.
		:return: List of loss values for each epoch.
		"""
		cost = []
		for epoch in range(epochs):
			# Forward pass
			yhat = self.forward(x)
			# Backward pass: compute gradients
			negative_gradient_weights, negative_gradient_bias = self.backward(x, yhat, y)
			# Update parameters
			self.weights += learning_rate * negative_gradient_weights
			self.bias += learning_rate * negative_gradient_bias
			# Compute new predictions and current loss
			yhat = self.forward(x)
			current_loss = self.loss_function(yhat, y)
			print(f"Epoch: {epoch + 1} | Mean Squared Error: {current_loss:.4f}")
			cost.append(current_loss.item())
		return cost


def load_data(csv_file: str):
	"""
	Load data from a CSV file. Assumes the CSV has two columns:
	first column for input features and second column for targets.

	:param csv_file: Path to the CSV file.
	:return: Tuple of torch.Tensor (x, y).
	"""
	# Load data assuming no header and two columns
	df= pd.read_csv(csv_file)
	# Reshape the first column to (samples, 1) and second column to (samples,)
	x = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float)
	y = torch.tensor(df['y'].values, dtype=torch.float)
	return x, y


def main():
	# Load CSV data from linreg-data.csv
	csv_file = "linreg-data.csv"
	try:
		x, y = load_data(csv_file)
	except Exception as e:
		print(f"Error loading CSV file: {e}")
		return

	# Initialize the regression model with determined number of features (assumed 1)
	model = ManualLinearRegression(number_of_features=x.shape[1], seed=42)

	# Train the model
	epochs = 50
	learning_rate = 0.01
	loss_history = model.train(x, y, epochs=epochs, learning_rate=learning_rate)

	# Plot loss history
	plt.figure(figsize=(8, 4))
	plt.plot(loss_history, label="Training Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Mean Squared Error")
	plt.title("Training Loss Over Epochs")
	plt.legend()
	plt.show()

	# Visualize the regression line compared to the data
	plt.figure(figsize=(8, 4))
	plt.scatter(x.numpy(), y.numpy(), label="Data", color="blue")
	predictions = model.forward(x).detach().numpy()
	plt.plot(x.numpy(), predictions, label="Regression Line", color="red")
	plt.xlabel("Input Feature")
	plt.ylabel("Target")
	plt.title("Manual Linear Regression Fit")
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()