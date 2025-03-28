"""
This script implements two versions of a linear regression model using PyTorch:
 - ManualLinearRegression (updates using manually computed gradients)
 - SemiManualLinearRegression (updates using torch.autograd.grad)

This updated version fixes the error:
  "element 0 of tensors does not require grad and does not have a grad_fn"
in the SemiManualLinearRegression by ensuring that the parameters require gradients.
"""

import torch
import torch.nn.functional as F
from torch.autograd import grad
import pandas as pd

# Dataset preparation
df = pd.read_csv('linreg-data.csv', index_col=0)
X = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float)
y = torch.tensor(df['y'].values, dtype=torch.float)

# Add constant to target variable (144)
y += 144

torch.manual_seed(42)

# Shuffle preparation data
shuffle_idx = torch.randperm(y.shape[0], dtype=torch.long)
X, y = X[shuffle_idx], y[shuffle_idx]

# Splitting the data into 70:30 ratio
percent_70 = int(shuffle_idx.size(0) * 0.7)

X_train, X_test = X[shuffle_idx[:percent_70]], X[shuffle_idx[percent_70:]]
y_train, y_test = y[shuffle_idx[:percent_70]], y[shuffle_idx[percent_70:]]

# Normalize the training data and test data
mu, sigma = X_train.mean(dim=0), X_train.std(dim=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma


# Manual Linear Regression Model
class ManualLinearRegression:
	def __init__(self, number_of_features: int, seed: int) -> None:
		self.number_of_features = number_of_features
		# parameters don't require grad since we manage the backward pass manually
		self.weights = torch.zeros(number_of_features, 1, dtype=torch.float)
		self.bias = torch.zeros(1, dtype=torch.float)
		torch.manual_seed(seed)

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		# weighted_sum = X * W + b
		weighted_sum = torch.add(torch.mm(X, self.weights), self.bias)
		activation = F.relu(weighted_sum)
		return activation.view(-1)

	def backward(self, X: torch.Tensor, yhat: torch.Tensor, y: torch.Tensor) -> (torch.Tensor, torch.Tensor):
		# Compute derivative of loss w.r.t predictions: derivative for (yhat - y)^2 is 2*(yhat - y)
		gradient_loss_yhat = 2 * (yhat - y)

		# Gradient with respect to weights
		gradient_loss_weights = torch.mm(X.T, gradient_loss_yhat.view(-1, 1)) / y.shape[0]
		# Gradient with respect to bias
		gradient_loss_bias = torch.sum(gradient_loss_yhat) / y.shape[0]

		# Return negative gradients for gradient descent
		return -gradient_loss_weights, -gradient_loss_bias

	def loss_function(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		return torch.mean((yhat - y) ** 2)

	def train(self, X: torch.Tensor, y: torch.Tensor, epochs: int, learning_rate: float = 0.01) -> list:
		cost = []

		for epoch in range(epochs):
			# Forward pass
			yhat = self.forward(X)
			# Backward pass: compute negative gradients
			neg_grad_w, neg_grad_b = self.backward(X, yhat, y)
			# Update parameters
			self.weights += learning_rate * neg_grad_w
			self.bias += learning_rate * neg_grad_b
			# Recalculate predictions and loss
			yhat = self.forward(X)
			current_loss = self.loss_function(yhat, y)
			print(f"Manual Regression - Epoch: {epoch + 1} | Mean Squared Error: {current_loss:.4f}")
			cost.append(current_loss.item())

		return cost


# Semi-Manual Linear Regression Model
class SemiManualLinearRegression:
	def __init__(self, number_of_features: int, seed: int) -> None:
		self.number_of_features = number_of_features
		# Initialize parameters with requires_grad=True for autograd to track them
		self.weights = torch.zeros(number_of_features, 1, dtype=torch.float, requires_grad=True)
		self.bias = torch.zeros(1, dtype=torch.float, requires_grad=True)
		torch.manual_seed(seed)

	def forward(self, X: torch.Tensor) -> torch.Tensor:
		# weighted_sum = X * W + b
		weighted_sum = torch.add(torch.mm(X, self.weights), self.bias)
		activation = F.relu(weighted_sum)
		return activation.view(-1)

	def loss_function(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		return torch.mean((yhat - y) ** 2)

	def train(self, X: torch.Tensor, y: torch.Tensor, epochs: int, learning_rate: float = 0.01) -> list:
		cost = []

		for epoch in range(epochs):
			# Forward pass
			yhat = self.forward(X)
			current_loss = self.loss_function(yhat, y)

			# Backward pass using torch.autograd.grad; computing gradients for weights and bias
			negative_gradient_weights = grad(current_loss, self.weights, retain_graph=True)[0] * -1
			negative_gradient_bias = grad(current_loss, self.bias)[0] * -1

			# Update parameters using a no_grad block to prevent tracking these operations
			with torch.no_grad():
				self.weights += learning_rate * negative_gradient_weights
				self.bias += learning_rate * negative_gradient_bias

			# Recalculate loss after update (no gradient tracking needed)
			yhat = self.forward(X)
			current_loss = self.loss_function(yhat, y)
			print(f"Semi-Manual Regression - Epoch: {epoch + 1} | Mean Squared Error: {current_loss:.4f}")
			cost.append(current_loss.item())

		return cost


if __name__ == "__main__":
	# Train Manual Linear Regression
	# print("Training Manual Linear Regression Model")
	# manual_model = ManualLinearRegression(number_of_features=X_train.shape[1], seed=42)
	# manual_cost = manual_model.train(X=X_train, y=y_train, epochs=50, learning_rate=0.01)

	# Train Semi-Manual Linear Regression
	print("\nTraining Semi-Manual Linear Regression Model")
	semi_manual_model = SemiManualLinearRegression(number_of_features=X_train.shape[1], seed=42)
	semi_manual_cost = semi_manual_model.train(X=X_train, y=y_train, epochs=50, learning_rate=0.01)

	# Optionally, you can add evaluation on the test data or plotting of the cost curves.
	print("\nTraining complete.")