#!/usr/bin/env python
"""
This script demonstrates two implementations of a simple linear regression model with a ReLU activation
using PyTorch. The two implementations are:
  1. Manual gradient computation (with correct ReLU and backward pass)
  2. Automatic differentiation with PyTorch's autograd

It reads a dataset from 'datasets/linreg-data.csv', converts all data to numeric,
drops any rows with missing values, and adds 144 to the target variable.
The script then shuffles the data, splits it into training and test sets,
trains both models, and prints out the loss per epoch.
"""

import torch
import torch.nn.functional as F
import pandas as pd

# ---------------------------
# Dataset Preparation
# ---------------------------
# Read CSV, ensure numeric conversion and drop rows with NaN values
df = pd.read_csv('linreg-data.csv', index_col=0, header=None)
df.columns = ['x1', 'x2', 'y']
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)  # Remove any rows with NaN values

# Add 144 to target variable 'y'
df['y'] += 144

# Convert to torch tensors
X = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float)
y = torch.tensor(df['y'].values, dtype=torch.float)

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

# Shuffle & train/test split
torch.manual_seed(42)
shuffle_idx = torch.randperm(y.shape[0])
X, y = X[shuffle_idx], y[shuffle_idx]
percent70 = int(shuffle_idx.size(0) * 0.7)
X_train, X_test = X[:percent70], X[percent70:]
y_train, y_test = y[:percent70], y[percent70:]


# ---------------------------
# Manual Linear Regression with ReLU Activation
# ---------------------------
class ManualLinearRegression:
	def __init__(self, number_of_features):
		"""
		Initialize weights and bias using zeros.
		We will update them manually using computed gradients.
		"""
		self.number_of_features = number_of_features
		# For random initialization using torch.rand, uncomment if desired:
		# self.weights = torch.rand(number_of_features, 1, dtype=torch.float)
		# self.bias = torch.rand(1, dtype=torch.float)
		self.weights = torch.zeros(number_of_features, 1, dtype=torch.float)
		self.bias = torch.zeros(1, dtype=torch.float)

	def forward(self, x):
		"""
		Forward pass with ReLU activation.
		Saves the weighted sum for use in the backward pass.
		:param x: input features tensor
		:return: activated output (ReLU applied) flattened to 1D
		"""
		weighted_sum = torch.add(x @ self.weights, self.bias)
		self.weighted_sum = weighted_sum  # save for backward (needed for ReLU derivative)
		activations = weighted_sum # F.relu(weighted_sum)
		return activations.view(-1)

	def backward(self, x, y, yhat):
		"""
		Compute gradients for weights and bias manually.
		Accounts for the derivative of ReLU.
		:param x: input mini-batch
		:param y: true labels mini-batch
		:param yhat: predictions from forward pass (after ReLU activation)
		:return: negative gradients for weights and bias
		"""
		# Loss: mean squared error => d(loss)/d(yhat) = 2*(y - yhat)
		gradient_loss_yhat = 2 * (y - yhat)
		# ReLU derivative: 1 if weighted_sum > 0, else 0
		relu_derivative = (self.weighted_sum > 0).float().view(-1)
		# Chain rule: d(loss)/d(weighted_sum) = d(loss)/d(yhat) * d(yhat)/d(weighted_sum)
		delta = gradient_loss_yhat * relu_derivative

		# Compute gradients for weights and bias
		gradient_loss_weights = torch.mm(x.t(), delta.view(-1, 1)) / y.shape[0]
		gradient_loss_bias = torch.sum(delta) / y.shape[0]

		# Return negative gradients for gradient ascent style update (weights += learning_rate * (-grad))
		return -gradient_loss_weights, -gradient_loss_bias


def loss_function(yhat, y):
	"""
	Calculate mean squared error loss.
	"""
	return torch.mean((yhat - y) ** 2)


def train_manual(model, x, y, epochs: int, learning_rate: float = 0.01, seed: int = 42, minibatch_size: int = 10):
	"""
	Train the manual linear regression model with computed gradients.
	Shuffles data and splits into mini-batches.
	:return: list of loss values per epoch
	"""
	cost = []
	torch.manual_seed(seed)

	for epoch in range(epochs):
		shuffle_idx = torch.randperm(y.shape[0])
		mini_batches = torch.split(shuffle_idx, minibatch_size)

		for mini_batch in mini_batches:
			yhat = model.forward(x[mini_batch])
			neg_grad_w, neg_grad_b = model.backward(x=x[mini_batch], y=y[mini_batch], yhat=yhat)

			model.weights += learning_rate * neg_grad_w
			model.bias += learning_rate * neg_grad_b

		yhat = model.forward(x)
		current_loss = loss_function(yhat, y)
		print(f'[Manual] Epoch: {epoch + 1:03d} | MSE: {current_loss.item():.5f}')
		cost.append(current_loss.item())
	return cost


# ---------------------------
# Automatic Linear Regression with PyTorch Autograd and ReLU
# ---------------------------
class AutoLinearRegression(torch.nn.Module):
	def __init__(self, number_of_features):
		"""
		Initialize using PyTorch's built-in Linear layer.
		We zero initialize the parameters.
		"""
		super(AutoLinearRegression, self).__init__()
		self.linear = torch.nn.Linear(in_features=number_of_features, out_features=1)
		self.linear.weight.detach().zero_()
		self.linear.bias.detach().zero_()

	def forward(self, x):
		"""
		Forward pass with ReLU activation.
		:param x: input features tensor
		:return: activated output flattened to 1D
		"""
		out = self.linear(x)
		return F.relu(out).view(-1)


def train_auto(model, x, y, epochs: int, learning_rate: float = 0.01, seed: int = 42, minibatch_size: int = 10):
	"""
	Train the model using PyTorch autograd.
	Uses manual mini-batch loop and updates parameters using gradient descent.
	"""
	cost = []
	torch.manual_seed(seed)
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

	for epoch in range(epochs):
		shuffle_idx = torch.randperm(y.size(0))
		mini_batches = torch.split(shuffle_idx, minibatch_size)

		for mini_batch in mini_batches:
			optimizer.zero_grad()
			yhat = model.forward(x[mini_batch])
			loss = F.mse_loss(yhat, y[mini_batch])
			loss.backward()
			optimizer.step()

		with torch.no_grad():
			yhat_full = model.forward(x)
			current_loss = loss_function(yhat_full, y)
			print(f'[Auto] Epoch: {epoch + 1:03d} | MSE: {current_loss.item():.5f}')
			cost.append(current_loss.item())

	return cost


# ---------------------------
# Running the training for both models
# ---------------------------
if __name__ == "__main__":
	print("\nTraining ManualLinearRegression Model:")
	manual_model = ManualLinearRegression(number_of_features=X_train.shape[1])
	_ = train_manual(manual_model, X_train, y_train, epochs=20, learning_rate=0.01, minibatch_size=10)

	print("\nTraining AutoLinearRegression Model:")
	auto_model = AutoLinearRegression(number_of_features=X_train.shape[1])
	_ = train_auto(auto_model, X_train, y_train, epochs=100, learning_rate=0.01, minibatch_size=10)

	print("\nTraining complete.")

# End of script
