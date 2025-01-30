import numpy as np
import matplotlib.pyplot as plt
import torch

data = np.genfromtxt('perceptron_toydata.txt', delimiter='\t')
X, y = data[:, :2], data[:, 2]
y = y.astype(int)

print('Class label counts:', np.bincount(y))
print('X.shape:', X.shape)
print('y.shape:', y.shape)

# Shuffling & train/test split
shuffle_idx = np.arange(y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, y = X[shuffle_idx], y[shuffle_idx]

X_train, X_test = X[shuffle_idx[:70]], X[shuffle_idx[70:]]
y_train, y_test = y[shuffle_idx[:70]], y[shuffle_idx[70:]]

# Normalize (mean zero, unit variance)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

device = torch.device("mps:0" if torch.cuda.is_available() else "cpu")


class Perceptron:
	def __init__(self, num_features):
		self.num_features = num_features
		self.weights = torch.zeros(num_features, 1,
								   dtype=torch.float32, device=device)
		self.bias = torch.zeros(1, dtype=torch.float32, device=device)

		# placeholder vectors so they don't
		# need to be recreated each time
		self.ones = torch.ones(1)
		self.zeros = torch.zeros(1)

	def forward(self, x):
		linear = torch.mm(x, self.weights) + self.bias
		predictions = torch.where(linear > 0., self.ones, self.zeros)
		return predictions

	def backward(self, x, y):
		predictions = self.forward(x)
		errors = y - predictions
		return errors

	# def train(self, x, y, epochs):
	#
	# 	__batch_size = 10
	# 	__batches = x.shape[0] // __batch_size
	#
	# 	for e in range(epochs):
	# 		# delta_weights = torch.zeros(self.num_features, 1, dtype=torch.float32, device=device)
	# 		# delta_bias = torch.zeros(1, dtype=torch.float32, device=device)
	# 		#
	# 		# # updated to batch mode
	# 		# for i in range(y.shape[0]):
	# 		#     # use view because backward expects a matrix (i.e., 2D tensor)
	# 		#     errors = self.backward(x[i].reshape(1, self.num_features), y[i]).reshape(-1)
	# 		#     delta_weights += (errors * x[i]).reshape(self.num_features, 1)
	# 		#     delta_bias += errors
	# 		#
	# 		# self.weights += delta_weights
	# 		# self.bias += delta_bias
	#
	# 		for batch in range(__batches):
	# 			start_idx = batch * __batch_size
	# 			end_idx = start_idx + __batch_size
	#
	# 			x_batch = x[start_idx:end_idx]
	# 			y_batch = y[start_idx:end_idx]
	#
	# 			delta_weights = torch.zeros(self.num_features, 1, dtype=torch.float32, device=device)
	# 			delta_bias = torch.zeros(1, dtype=torch.float32, device=device)
	#
	# 			for i in range(y_batch.shape[0]):
	# 				errors = self.backward(x_batch[i].reshape(1, self.num_features), y_batch[i]).reshape(-1)
	#
	# 				delta_weights += (errors * x_batch[i]).reshape(self.num_features, 1)
	# 				delta_bias += errors
	#
	# 			self.weights += delta_weights
	# 			self.bias += delta_bias

	def train(self, x, y, epochs, batch_size=10):
		"""
		Train the perceptron using mini-batch gradient descent.

		Args:
			x: Input features tensor
			y: Target labels tensor
			epochs: Number of training epochs
			batch_size: Size of mini-batches (default: 32)
		"""
		n_samples = x.shape[0]
		indices = torch.arange(n_samples, device=device)

		for _ in range(epochs):
			# Shuffle indices at the start of each epoch
			shuffled_indices = indices[torch.randperm(n_samples, device=device)]

			# Process mini-batches
			for start_idx in range(0, n_samples, batch_size):
				batch_indices = shuffled_indices[start_idx:start_idx + batch_size]
				x_batch = x[batch_indices]  # x[start_idx:start_idx + batch_size]
				y_batch = y[batch_indices]  # y[start_idx:start_idx + batch_size]

				# Forward pass and compute errors for entire batch
				predictions = self.forward(x_batch)
				errors = (y_batch - predictions.reshape(-1)).reshape(-1, 1)

				# Compute weight and bias updates for entire batch at once
				delta_weights = torch.mm(x_batch.t(), errors)
				delta_bias = torch.sum(errors)

				# Update parameters
				self.weights += delta_weights
				self.bias += delta_bias

	def evaluate(self, x, y):
		predictions = self.forward(x).reshape(-1)
		accuracy = torch.sum(predictions == y).float() / y.shape[0]
		return accuracy


ppn = Perceptron(num_features=2)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

ppn.train(X_train_tensor, y_train_tensor, epochs=5)

print('Model parameters:')
print('  Weights: %s' % ppn.weights)
print('  Bias: %s' % ppn.bias)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

test_acc = ppn.evaluate(X_test_tensor, y_test_tensor)
print('Test set accuracy: %.2f%%' % (test_acc * 100))
