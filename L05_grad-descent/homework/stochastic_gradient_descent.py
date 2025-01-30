"""
Implement the stochastic gradient descent (SGD) algorithm for the linear regression model using pytorch.
"""

import torch

class LinearRegression:

	def __init__(self):
		# what do need?
		self.weights = torch.zeros(1, )
		self.bias = torch.zeros(0)

