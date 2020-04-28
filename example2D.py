import matplotlib.pyplot as plt
import torch
from LinearRegressor import LinearRegressor

# Creating synthetic data points
angular_coef = 2
linear_coef = 1
n_points = 150
x_data = 2*torch.rand(n_points)
y_data = angular_coef*x_data + linear_coef + 1.5*torch.rand((n_points))
data = torch.empty((n_points, 2))
data[:, 0] = x_data
data[:, 1] = y_data

# Fitting regressor
iterations = 50
regressor = LinearRegressor()
regressor.fit(data, n_iters=iterations)

# Creating subplots
plt.style.use('ggplot')
fig, ax = plt.subplots(2, 1)
plt.suptitle("2D Linear Regression Example", fontsize=16)

# Plotting synthetic data and fitted line
ax[0].scatter(x_data, y_data, s=6)
x_line = torch.Tensor([[0], [2]])
y_line = regressor.predict(x_line)
ax[0].plot(x_line, y_line, color='black')
ax[0].set_ylabel("feature 2")
ax[0].set_xlabel("feature 1")

# Plotting the error during fitting process
ax[1].plot(range(1, iterations+1), regressor.get_error(), color='blue')
ax[1].set_ylabel("Mean Squared Error")
ax[1].set_xlabel("Iterations")

plt.show()

