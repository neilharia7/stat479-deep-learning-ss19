import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class Constants:
	NUMBER_OF_FEATURES = 4
	NUMBER_OF_CLASSES = 3
	LEARNING_RATE = 0.05
	EPOCHS = 100


class IrisDataset(Dataset):
	def __init__(self, data, labels):
		self.data = torch.tensor(data, dtype=torch.float32)
		self.labels = torch.tensor(labels, dtype=torch.long)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]


iris = load_iris()
data = iris.data
labels = iris.target

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=12)
train_dataset = IrisDataset(X_train, y_train)
test_dataset = IrisDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


class LogisticRegression(nn.Module):
	def __init__(self, number_of_features, number_of_classes):
		super().__init__()
		self.linear = nn.Linear(number_of_features, number_of_classes)

	def forward(self, x):
		linear = self.linear(x)
		probas = torch.sigmoid(linear)
		return linear, probas


model = LogisticRegression(
	number_of_features=Constants.NUMBER_OF_FEATURES, number_of_classes=Constants.NUMBER_OF_CLASSES)
cross_entropy_loss = nn.CrossEntropyLoss()
sgd_optimizer = torch.optim.SGD(model.parameters(), lr=Constants.LEARNING_RATE)

train_loss = []
for epoch in range(Constants.EPOCHS):

	current_loss = 0
	for inputs, labels in train_loader:
		# calc net inputs
		outputs, _ = model(inputs)

		loss = cross_entropy_loss(outputs, labels)

		sgd_optimizer.zero_grad()
		loss.backward()
		sgd_optimizer.step()

		current_loss += loss.item()

	train_loss.append(current_loss / len(train_loader))
	print(f"Epoch [{epoch + 1}/{Constants.EPOCHS}], Loss: {current_loss / len(train_loader):.4f}")

# Plot loss over epochs
plt.plot(train_loss, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


model.eval()
with torch.no_grad():
	correct = 0
	total = 0
	for inputs, labels in test_loader:
		outputs, _ = model(inputs)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	print(f"Test Accuracy: {100 * correct / total:.2f}%")
