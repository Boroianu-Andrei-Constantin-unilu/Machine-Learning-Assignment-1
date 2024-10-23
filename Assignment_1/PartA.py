import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import *

# Reading the file is necessary for data analysis
powerplant_data = pd.read_csv('powerplant.csv')

# Task A.I.1
# Assuming the group number is 19, extract 20 data points, starting from index 20 * 19
n = 19
start_index = 20 * n
end_index = start_index + 20
subset_data = powerplant_data.iloc[start_index:end_index]

# Plot the extracted 20 data points
plt.scatter(subset_data['x'], subset_data['y'], color='blue')
plt.title('Ambient Temperature vs Energy Output')
plt.xlabel('Ambient Temperature (Celsius Degrees)')
plt.ylabel('Energy Output (MegaWatts)')
plt.grid(True)
plt.show()

# Task A.I.2
# Normalize the data, using z-scores (mean = 0, std = 1)
scaler = StandardScaler()
normalized_data = scaler.fit_transform(subset_data)

# Conversion back to DataFrame, to ease the use
normalized_df = pd.DataFrame(normalized_data, columns=['x_normalized', 'y_normalized'])

# Display of normalized data
print(normalized_df.head())

# Task A.I.3
# Extractions of normalized x and y values
X = normalized_df['x_normalized']
Y = normalized_df['y_normalized']

# Adding columns of 1 to x, for the intercept term
X_matrix = np.vstack([X, np.ones(len(X))]).T

# Closed-form solution for linear regression: theta = (X^T X)^-1 X^T Y
theta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ Y

# Extract of coefficients and compute the predicted Y values, based on the linear model
a, b = theta
predicted_Y = a * X + b

# Plot the regression line against the data points, and displaying the coefficients
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, predicted_Y, color='red', label=f'Regression Line: Y = {a:.2f}X + {b:.2f}')
plt.title('Linear Regression on Normalized Data')
plt.xlabel('Normalized Ambient Temperature')
plt.ylabel('Normalized Energy Output')
plt.legend()
plt.grid(True)
plt.show()

print (a, b)

# Task A.I.4
# Definition of the MSE (Mean Squared Error) function
def mse_cost(a, b, X, Y):
    return np.mean((Y - (a * X + b)) ** 2)

# Calculation of derivatives of the cost function
def compute_gradients(a, b, X, Y):
    n = len(X)
    da = (-2/n) * np.sum(X * (Y - (a * X + b)))
    db = (-2/n) * np.sum(Y - (a * X + b))
    return da, db

# Task A.I.5
# Set arbitrary initial values for a and b and compute the initial cost
a_init = 0
b_init = 0
initial_cost = mse_cost(a_init, b_init, X, Y)

# Perform iteration of gradient descent and update parameters
learning_rate = 0.01
da, db = compute_gradients(a_init, b_init, X, Y)

a_new = a_init - learning_rate * da
b_new = b_init - learning_rate * db

# Compute the new costs after one iteration, then display the results
new_cost = mse_cost(a_new, b_new, X, Y)
print(initial_cost, new_cost, a_new, b_new)

# Task A.I.6
# Define function to perform gradient descent
def gradient_descent(X, Y, learning_rate = 0.01, iterations = 100):
    a = 0
    b = 0
    costs = []
    
    for i in range(iterations):
        da, db = compute_gradients(a, b, X, Y)
        a -= learning_rate * da
        b -= learning_rate * db
        costs.append(mse_cost(a, b, X, Y))
        
    return a, b, costs

# Perform gradient descent with a small learning rate
# Track the cost over iterations
learning_rate = 0.05
iterations = 100
a_final, b_final, costs = gradient_descent(X, Y, learning_rate, iterations)

# Plot the cost functions over iterations and print final parameters after gradient descent
plt.plot(range(iterations), costs)
plt.title('Cost Function Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.grid(True)
plt.show()

print(a_final, b_final)

# Task A.II.1
# Define the cost function for polynomial regression
def polynomial_cost(theta0, theta1, theta2, X, Y):
    n = len(X)
    predictions = theta2 * X**2 + theta1 * X + theta0
    return np.sum((Y - predictions) ** 2) / (4 * n)

# Define gradient of cost function, with respect to the theta variables
def polynomial_gradients(theta0, theta1, theta2, X, Y):
    n = len(X)
    error = (Y - (theta2 * X**2 + theta1 * X + theta0))
    d_theta0 = (-2/n) * np.sum(error)
    d_theta1 = (-2/n) * np.sum(X * error)
    d_theta2 = (-2/n) * np.sum(X**2 * error)
    return d_theta0, d_theta1, d_theta2

# Define function to perform the gradient descent for polynomial regression
def gradient_descent_polynomial(X, Y, learning_rate = 0.01, iterations = 100):
    theta0 = 0
    theta1 = 0
    theta2 = 0
    costs = []
    
    for i in range(iterations):
        d_theta0, d_theta1, d_theta2 = polynomial_gradients(theta0, theta1, theta2, X, Y)
        theta0 -= learning_rate * d_theta0
        theta1 -= learning_rate * d_theta1
        theta2 -= learning_rate * d_theta2
        costs.append(polynomial_cost(theta0, theta1, theta2, X, Y))
        
    return theta0, theta1, theta2, costs

# Task A.II.2
# Perform gradient descent, for polynomial regression
# Track the cost over iterations
learning_rate = 0.05
iterations = 100
theta0_final, theta1_final, theta2_final, poly_costs = gradient_descent_polynomial(X, Y, learning_rate, iterations)

# Plot the cost functions for polynomial regression and print final polynomial coefficients
plt.plot(range(iterations), poly_costs)
plt.title('Polynomial Regression: Cost Function Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.grid(True)
plt.show()

print(theta0_final, theta1_final, theta2_final)

# Plot the polynomial regression curve, along with the original data points
X_sorted = np.sort(X) # X is sorted for a smoother plot
Y_poly_pred = theta2_final * X_sorted**2 + theta1_final * X_sorted + theta0_final

plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X_sorted, Y_poly_pred, color='green', label='Polynomial Regression')
plt.plot(X_sorted, a_final * X_sorted + b_final, color='red', label='Linear Regression')
plt.title('Polynomial vs Linear Regression')
plt.xlabel('Normalized Ambient Temperature')
plt.ylabel('Normalized Energy Output')
plt.legend()
plt.grid(True)
plt.show()