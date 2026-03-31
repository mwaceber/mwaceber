import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import sympy as sp

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        #Initialize weights by Xavier/Glorot which is better for sigmoid
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/ input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/ hidden_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]
        #Include sigmoid derivative for output layer
        self.dz2 = (output - y) * output * (1 - output)
        self.dw2 = np.dot(self.a1.T, self.dz2) / m
        self.db2 = np.sum(self.dz2, axis=0, keepdims= True)/ m

        #Hidden layer
        self.dz1 = np.dot(self.dz2, self.W1.T) * self.a1 * (1 - self.a1)
        self.dw1 = np.dot(X.T, self.dz1) / m
        self.db1 = np.sum(self.dz1, axis=0, keepdims=True)/ m 

    def train(self, X, y, epochs = 10000, learning_rate = 0.01):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y)**2)
            losses.append(loss)

            self.backward(X, y, output)

            self.W1 -= learning_rate * self.dw1
            self.b1 -= learning_rate * self.db1
            self.W2 -= learning_rate * self.dw2
            self.b2 -= learning_rate * self.db2

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss: .6f}")
        return losses

#Create synthetic datasets for binary classification
np.random.seed(42)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

#Test a fixed version of a nueral network
nn = NeuralNetwork( input_size= 2, hidden_size= 4, output_size=1)
losses = nn.train(X, y, epochs= 10000, learning_rate= 0.5)

print("\nFinal predictions:")
for i in range(len(X)):
    pred = nn.forward(X[i: i+1])
    print(f"Input: {X[i]}, Predicted: {pred[0][0]:.4f}, Actual: {y[1][0]}")