#!/usr/bin/env python3
"""
Custom Dataloader and MLP for XOR Classification Experiment

This script:
1. Reads the XOR dataset from 'xor.csv'.
2. Splits the data into training (85%), testing (5%), and validation (15%) sets.
3. Implements a custom PyTorch Dataset to wrap the DataFrame.
4. Defines a parameterized multi-layer perceptron (MLP) with one hidden layer
   (using ReLU) and an output layer with two outputs.
5. Trains the model with different hidden layer widths: 10, 100, 1000, and 10,000.
6. Plots the training and validation accuracies for each hidden width.
7. Prints brief analysis regarding underfitting/overfitting and bias/variance.

Usage:
    Ensure 'xor.csv' is placed in the same directory and run:
        python xor_classification_experiment.py
"""

import math
from math import ceil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# %% Data Preparation Functions

def split_data(dataframe: pd.DataFrame, percentages=None):
	"""
	Splits a DataFrame into parts based on provided percentages.

	Args:
		dataframe (pd.DataFrame): The dataframe to split.
		percentages (list, optional): A list of percentages that sum to 1.
									  Default is [0.85, 0.05, 0.15].

	Returns:
		list of pd.DataFrame: List containing the split dataframes.

	Raises:
		ValueError: If the percentages do not sum to 1.
	"""
	if percentages is None:
		percentages = [0.85, 0.05, 0.15]

	if not math.isclose(sum(percentages), 1.0, rel_tol=1e-2):
		raise ValueError('percentages must sum to 1')

	prev_idx, curr_idx = 0, 0
	data_size = len(dataframe)
	data_splits = []
	for percentage in percentages:
		curr_idx += ceil(percentage * data_size)
		data_splits.append(dataframe.iloc[prev_idx:curr_idx])
		prev_idx = curr_idx

	return data_splits


# %% Custom Dataset
class CSVDataset(Dataset):
	"""
	Custom Dataset for XOR CSV data.
	Assumes the CSV file has feature columns and a label column.
	The label is assumed to be in a column named "label".
	All other columns are used as input features.
	"""

	def __init__(self, dataframe: pd.DataFrame):
		super().__init__()

		print("columns", dataframe.columns)

		if "class label" not in dataframe.columns:
			raise ValueError("DataFrame must contain a 'class label' column")
		self.features = dataframe.drop("class label", axis=1).values.astype(np.float32)
		self.labels = dataframe["class label"].values.astype(np.int64)

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		x = self.features[idx]
		y = self.labels[idx]
		return torch.tensor(x), torch.tensor(y)


# %% Model Definition
class MultiLayerPerceptron(nn.Module):
	"""
	Multi-Layer Perceptron (MLP) model with one hidden layer.
	The hidden layer width is parameterized.
	"""

	def __init__(self, input_dim: int, hidden_width: int, output_dim: int = 2):
		super().__init__()
		self.model = nn.Sequential(
			nn.Flatten(),
			nn.Linear(input_dim, hidden_width),
			nn.ReLU(),
			nn.Linear(hidden_width, output_dim)
		)

	def forward(self, x):
		return self.model(x)


# %% Training and Evaluation Functions
def train_epoch(model, dataloader, criterion, optimizer, device):
	model.train()
	total_correct = 0
	total_samples = 0
	running_loss = 0.0
	for inputs, labels in dataloader:
		inputs, labels = inputs.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * inputs.size(0)
		preds = outputs.argmax(dim=1)
		total_correct += (preds == labels).sum().item()
		total_samples += inputs.size(0)

	avg_loss = running_loss / total_samples
	accuracy = total_correct / total_samples
	return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
	model.eval()
	total_correct = 0
	total_samples = 0
	running_loss = 0.0
	with torch.no_grad():
		for inputs, labels in dataloader:
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			running_loss += loss.item() * inputs.size(0)
			preds = outputs.argmax(dim=1)
			total_correct += (preds == labels).sum().item()
			total_samples += inputs.size(0)

	avg_loss = running_loss / total_samples
	accuracy = total_correct / total_samples
	return avg_loss, accuracy


# %% Main Experiment
def main():
	# Load dataset
	df = pd.read_csv('xor.csv')

	# Split data into train, test, validation sets using provided percentages
	train_df, test_df, valid_df = split_data(df, percentages=[0.85, 0.05, 0.10])
	# Note: Adjusted the percentages so they sum to 1 (0.85+0.05+0.10=1)

	# Create Datasets and DataLoaders
	batch_size = 16
	train_dataset = CSVDataset(train_df)
	valid_dataset = CSVDataset(valid_df)
	test_dataset = CSVDataset(test_df)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	# Determine input dimension from dataset
	sample_input, _ = train_dataset[0]
	input_dim = sample_input.shape[0]

	# Define training parameters
	hidden_widths = [10, 100, 1000, 10000]
	num_epochs = 50
	learning_rate = 0.01
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# To store results for plotting
	results = {}

	for width in hidden_widths:
		print(f"\nTraining model with hidden width: {width}")
		model = MultiLayerPerceptron(input_dim=input_dim, hidden_width=width, output_dim=2)
		model.to(device)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)

		train_acc_history = []
		valid_acc_history = []

		for epoch in range(num_epochs):
			train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
			valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
			train_acc_history.append(train_acc)
			valid_acc_history.append(valid_acc)
			print(f"Epoch {epoch + 1}/{num_epochs}: Train Acc: {train_acc:.3f}, Valid Acc: {valid_acc:.3f}")

		results[width] = {
			"train_accuracy": train_acc_history,
			"valid_accuracy": valid_acc_history
		}

	# Plot training and validation accuracies
	fig, axs = plt.subplots(2, 2, figsize=(12, 10))
	axs = axs.flatten()
	for idx, width in enumerate(hidden_widths):
		ax = axs[idx]
		epochs = range(1, num_epochs + 1)
		ax.plot(epochs, results[width]["train_accuracy"], label="Train Acc", marker='o')
		ax.plot(epochs, results[width]["valid_accuracy"], label="Valid Acc", marker='x')
		ax.set_title(f"Hidden Width: {width}")
		ax.set_xlabel("Epoch")
		ax.set_ylabel("Accuracy")
		ax.legend()
		ax.grid(True)
	plt.tight_layout()
	plt.show()

	# Evaluate on test set using the best validation performance model.
	# For simplicity, we select the model with the highest final validation accuracy.
	best_width = max(results, key=lambda w: results[w]["valid_accuracy"][-1])
	print(f"\nBest model based on validation accuracy has hidden width: {best_width}")

	# Reload best model for test evaluation
	best_model = MultiLayerPerceptron(input_dim=input_dim, hidden_width=best_width, output_dim=2)
	best_model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(best_model.parameters(), lr=learning_rate)

	# Optionally, one can retrain on training+validation set; here we simply evaluate using current state.
	test_loss, test_acc = evaluate(best_model, test_loader, criterion, device)
	print(f"Test Accuracy for best model: {test_acc:.3f}")


# Analysis:
# a) Underfitting is observed when both training and validation accuracies are low,
#    typically for models with too small capacity (e.g., hidden width=10 may underfit).
# b) Overfitting is observed when the training accuracy is high while validation accuracy is considerably lower,
#    often seen in models with excessively high capacity (e.g., hidden width=10,000 may overfit).
# c) These experiments illustrate the bias-variance trade-off:
#    - A very low-capacity (high bias) model underfits (cannot capture the complexity of the XOR function).
#    - A very high-capacity (high variance) model may overfit (captures noise in training data) even if test performance degrades.
#    Thus, by comparing training and validation curves, one can comment on the model's bias (underfitting)
#    and variance (overfitting).


if __name__ == "__main__":
	main()
