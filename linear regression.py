import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import sympy as sp
#Set random seed for reproducibility
np.random.seed(42)

class LinearRegression:
#Its core purpose is to initialize the model's hyperparameters and parameters
    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        #Learning rate controls how big each step is during the gradient descent
        #Also it scales the gradient in updates to minimize the loss function
        #Too high==Overshoot the minimum, too low==slow convergence
        #n_iteration is the no of times the gradient descent loop runs to update the parameters
        """Linear Regression model using Gradient Descent."""
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    #Weight is a vector of coefficients for each feature.Its the slope
    #Bias is the y-intercept
    #Loss history tracks errors in each iteration for monitoring training progress 
    def fit(self, X, y):
        """Train using gradient descent."""
        n_samples, n_features = X.shape #Extracts dimensions for calculations
        # Initialize parameters( Linear Algebra: Vector Initialization)
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent(Calculus: Finding minima)
        for i in range(self.n_iters):
            # Linea models predictions
            y_pred = np.dot(X, self.weights) + self.bias #(X.weights) computes the linear combinations of all samples at once
            # Compute loss (Mean Squared Error)
            loss = (1/n_samples) * np.sum((y_pred - y)**2)
            self.loss_history.append(loss)
            # Compute gradients (Partial derivatives)
            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))# Measures how much weight need to change to reduce loss
            db = (2/n_samples) * np.sum(y_pred - y)# Measures how much bias need to change to reduce loss
            
            # Update parameters
            self.weights -= self.lr * dw #Move oppposite to gradient for descent
            self.bias -= self.lr * db #Subtracts scaled gradients from bias 

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        return np.dot(X, self.weights) + self.bias
        #Uses trained weights and bias to predict y for the new X

# Generate sample data
X, y = make_classification(n_samples= 100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
y = y * 10 + 5 # Scale for suitable regression

#Train for model
model = LinearRegression(learning_rate=0.1, n_iterations=500)
model.fit(X, y)

#Plot loss convegence
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(model.loss_history)
#Shows if gradient descent is working properly
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Gradient Descent Convergence')

#Plot predictions vs actual
plt.subplot(1, 2, 2)
predictions = model.predict(X)
plt.scatter(y, predictions, alpha = 0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw = 2)
#Perfect predictions would lie on the red dashed line, so we can visually assess the model's performance.
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Predicted vs Actual')
plt.tight_layout()
plt.show()