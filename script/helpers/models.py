import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1(nn.Module):
	"""
	CNN with 2 convolutional layers (with maxpooling) and 2 fully connected layers.
	"""

	def __init__(self, num_classes=101):
		super(CNN1, self).__init__()
		# First convolutional layer
		self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Second convolutional layer
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Calculate the size after convolutions and pooling
		# Input: 224x224 -> conv1 -> 224x224 -> pool1 -> 112x112
		# -> conv2 -> 112x112 -> pool2 -> 56x56
		# So the feature map size is 56x56 with 64 channels
		self.fc1 = nn.Linear(64 * 56 * 56, 512)
		self.fc2 = nn.Linear(512, num_classes)

	def forward(self, x):
		x = self.pool1(F.relu(self.conv1(x)))
		x = self.pool2(F.relu(self.conv2(x)))
		x = x.view(-1, 64 * 56 * 56)  # Flatten
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x


class CNN2(nn.Module):
	"""
	CNN with 3 convolutional layers (with maxpooling) and 2 fully connected layers.
	"""

	def __init__(self, num_classes=101):
		super(CNN2, self).__init__()
		# First convolutional layer
		self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Second convolutional layer
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Third convolutional layer
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Calculate the size after convolutions and pooling
		# Input: 224x224 -> conv1 -> 224x224 -> pool1 -> 112x112
		# -> conv2 -> 112x112 -> pool2 -> 56x56
		# -> conv3 -> 56x56 -> pool3 -> 28x28
		# So the feature map size is 28x28 with 128 channels
		self.fc1 = nn.Linear(128 * 28 * 28, 512)
		self.fc2 = nn.Linear(512, num_classes)

	def forward(self, x):
		x = self.pool1(F.relu(self.conv1(x)))
		x = self.pool2(F.relu(self.conv2(x)))
		x = self.pool3(F.relu(self.conv3(x)))
		x = x.view(-1, 128 * 28 * 28)  # Flatten
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x


class CNN3(nn.Module):
	"""
	CNN with 4 convolutional layers (with maxpooling) and 2 fully connected layers.
	"""

	def __init__(self, num_classes=101):
		super(CNN3, self).__init__()
		# First convolutional layer
		self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Second convolutional layer
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Third convolutional layer
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Fourth convolutional layer
		self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Calculate the size after convolutions and pooling
		# Input: 224x224 -> conv1 -> 224x224 -> pool1 -> 112x112
		# -> conv2 -> 112x112 -> pool2 -> 56x56
		# -> conv3 -> 56x56 -> pool3 -> 28x28
		# -> conv4 -> 28x28 -> pool4 -> 14x14
		# So the feature map size is 14x14 with 256 channels
		self.fc1 = nn.Linear(256 * 14 * 14, 512)
		self.fc2 = nn.Linear(512, num_classes)

	def forward(self, x):
		x = self.pool1(F.relu(self.conv1(x)))
		x = self.pool2(F.relu(self.conv2(x)))
		x = self.pool3(F.relu(self.conv3(x)))
		x = self.pool4(F.relu(self.conv4(x)))
		x = x.view(-1, 256 * 14 * 14)  # Flatten
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x
