import torch as t
import torch.nn.functional as F



class LinearRegression:
	def __init__(self, number_of_features):
		self.number_of_features = number_of_features
		self.weights = t.zeros(number_of_features, 1, dtype=t.float32)
		self.biases = t.zeros(1, dtype=t.float32)

	def forward(self, x):
		y_hat = t.sum(t.mm(x, self.weights), self.biases)
		activation = F.relu(y_hat)
		return activation.view(-1)

	def backward(self, x, y_hat, y):
		gradient_loss_yhat = 2 * (y - y_hat)

		gradient_loss_weights = t.mm(x.T, gradient_loss_yhat.view(-1, 1)) / y.shape[0]

		gradient_loss_bias = t.sum(gradient_loss_yhat) / y.shape[0]

		return -gradient_loss_weights, -gradient_loss_bias

	def train(self, x, y, epochs, learning_rate):

		for e in range(epochs):

			y_hat = self.forward(x)

			negative_grad_w, negative_grad_b = self.backward(x, y_hat, y)
			self.weights += learning_rate * negative_grad_w
			self.biases += learning_rate * negative_grad_b
