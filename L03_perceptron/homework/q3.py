"""
Consider dataset `Data` generated using the python code below:

Visualize (using matplotlib) the above data and comment on what accuracy you expect the
perceptron learning algorithm to result in. Explain your reason.

Using the first 150 samples from class 1 (a1) and first 150 samples from class 0 (a2) for training,
and the rest of the samples for testing, train the perceptron for 10 epochs.
What accuracy do you get on training data? What accuracy do you get on testing data?


"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

a1 = np.random.uniform(4, 6, [200, 2])
a2 = np.random.uniform(0, 10, [200, 2])

Data_X = np.vstack((a1, a2))
Data_Y = np.hstack((np.ones(200).T, np.zeros(200).T)).astype(int)

# Sub question 1
# plt.figure(figsize=(8, 8))
# plt.scatter(a1[:, 0], a1[:, 1], c='red', label='Class 1')
# plt.scatter(a2[:, 0], a2[:, 1], c='blue', label='Class 0')
#
# plt.xlabel('Data X')
# plt.ylabel('Data Y')
# plt.legend()
# plt.grid(True)
# plt.show()

# Sub question 2
from sklearn.linear_model import Perceptron

# use 150 training samples
train_X = np.vstack((a1[:150], a2[:150]))
train_Y = np.hstack((np.ones(150).T, np.zeros(150).T)).astype(int)

# use 50 samples for testing
text_X = np.vstack((a1[150:], a2[150:]))
text_Y = np.hstack((np.ones(50).T, np.zeros(50).T)).astype(int)

perceptron = Perceptron(max_iter=10)  # 10 epochs
perceptron.fit(train_X, train_Y)

train_predictions = perceptron.predict(train_X)
# print("train_predictions: ", train_predictions)
train_accuracy = accuracy_score(train_Y, train_predictions)
print("train_accuracy: ", train_accuracy)

test_predictions = perceptron.predict(text_X)
# print("test_predictions: ", test_predictions)
test_accuracy = accuracy_score(text_Y, test_predictions)
print("test_accuracy: ", test_accuracy)