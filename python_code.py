import numpy as np
import math
import matplotlib.pyplot as plt

# values where taken Previously and given to the vector variable
vector = np.array(
    [
        [3.09, -40.64, -0.71],
        [3.09, -40.27, -0.61],
        [3.27, -40.27, -0.61],
        [3.36, -40.45, -0.61],
        [3.27, -40.45, -0.71],
        [3.36, -40.45, -0.71],
        [3.09, -40.55, -0.71],
        [3.27, -40.36, -0.92],
        [3.45, -40.64, -0.71],
        [3.45, -40.64, -0.82],
        [3.27, -40.45, -1.02],
        [3.18, -40.36, -0.92],
        [3.18, -40.64, -0.82],
        [3.36, -40.64, -0.71],
        [3.18, -40.55, -0.82],
        [3.27, -40.73, -0.61],
        [3.55, -40.36, -0.82],
        [3.27, -40.64, -0.92],
        [3.18, -40.55, -0.92],
        [3.36, -40.73, -0.92],
        [3.18, -40.55, -0.92],
        [3.45, -40.55, -0.71],
        [3.45, -40.64, -0.71],
        [3.55, -40.64, -0.82],
        [3.27, -40.55, -0.92],
        [3.36, -40.36, -0.71],
        [3.27, -40.45, -0.71],
        [3.18, -40.45, -0.61],
        [3.09, -40.64, -0.61],
        [3.27, -40.27, -0.92],
        [3.09, -40.45, -0.71],
        [3.36, -40.36, -0.82],
        [3.27, -40.55, -0.82],
        [3.45, -40.45, -0.41],
        [3.45, -40.55, -0.51],
        [3.36, -40.45, -0.92],
        [3.27, -40.55, -0.82],
        [3.18, -40.45, -1.02],
        [3.18, -40.55, -0.71],
        [3.55, -40.73, -0.92],
        [3.36, -40.82, -0.92],
        [3.27, -40.45, -0.71],
        [3.55, -40.64, -0.82],
        [3.36, -40.18, -0.71],
        [3.18, -40.64, -0.82],
        [3.36, -40.64, -0.92],
        [3.27, -40.64, -1.02],
        [3.27, -40.55, -0.92],
        [3.45, -40.82, -0.82],
        [3.27, -40.64, -1.02],
        [3.36, -40.55, -1.02],
        [3.09, -40.36, -1.12],
        [3.09, -40.45, -0.61],
        [3.45, -40.45, -0.41],
        [3.45, -40.55, -0.51],
        [3.36, -40.45, -0.92],
        [3.27, -40.55, -0.82],
        [3.18, -40.45, -1.02],
        [3.18, -40.55, -0.71],
        [3.55, -40.73, -0.92],
    ]
)
len(vector)

weights_input = np.array([0.2, 0.3, 0.1])
weights_n = np.array([0.4, 0.2, 0.5])
weights_o = np.array([0.8, 0.7, 0.6])
h = np.array([0, 0, 0])
output = np.empty((0, 3))
# Learning rate (controls the step size during weight updates)
learning_rate = 0.001

# Number of epochs for training
num_epochs = 60

# Training loop
for epoch in range(num_epochs):
    loss = 0.0
    for i in range(len(vector)):
        # Forward propagation
        mul = vector[i] * weights_input
        fun = weights_n * h + mul
        h0 = math.tanh(fun[0])
        h1 = math.tanh(fun[1])
        h2 = math.tanh(fun[2])
        h = np.array([h0, h1, h2])
        final = weights_o * h
        # Calculate the loss for this data point
        data_loss = np.mean((final - vector[i]) ** 2)
        loss += data_loss

        # Backward propagation (updating weights)
        d_output = 2.0 * (final - vector[i])
        d_hidden_activation = d_output * weights_o
        d_fun = d_hidden_activation * (1.0 - h**2)

        weights_n -= learning_rate * d_fun * h
        weights_input -= learning_rate * d_fun * vector[i]
        weights_o -= learning_rate * d_output * h

    # Normalize the loss by the number of data points
    loss /= len(vector)

    # Print the loss after each epoch
    print(f"Epoch {epoch + 1} - Loss: {loss:.6f}")


for i in range(len(vector)):
    mul = vector[i] * weights_input
    fun = weights_n * h + mul
    h0 = math.tanh(fun[0])
    h1 = math.tanh(fun[1])
    h2 = math.tanh(fun[2])
    h = np.array([h0, h1, h2])
    final = weights_o * h
    print("Prediction", i + 1, final)
    output = np.vstack((output, final))

print("\nOutput array:")
print(output)
print(weights_input)
print(weights_n)
print(weights_o)
fig, axes = plt.subplots(3, 1, figsize=(15, 6))

# Plotting X-dimension
axes[0].plot(range(len(vector)), vector[:, 0], label="Vector x", color="blue")
axes[0].plot(range(len(output)), output[:, 0], label="Output x", color="red")
axes[0].set_xlabel("Index")
axes[0].set_ylabel("X-dimension")
axes[0].legend()

# Plotting Y-dimension
axes[1].plot(range(len(vector)), vector[:, 1], label="Vector y", color="blue")
axes[1].plot(range(len(output)), output[:, 1], label="Output y", color="red")
axes[1].set_xlabel("Index")
axes[1].set_ylabel("Y-dimension")
axes[1].legend()

# Plotting Z-dimension
axes[2].plot(range(len(vector)), vector[:, 2], label="Vector z", color="blue")
axes[2].plot(range(len(output)), output[:, 2], label="Output z", color="red")
axes[2].set_xlabel("Index")
axes[2].set_ylabel("Z-dimension")
axes[2].legend()
plt.tight_layout()
plt.show()
error = output - vector
fig, axes = plt.subplots(3, 1, figsize=(15, 6))

# Plotting X-dimension
axes[0].plot(range(len(error)), error[:, 0], label="Vector x", color="blue")
axes[0].set_xlabel("Index")
axes[0].set_ylabel("X-dimension")
axes[0].legend()

# Plotting Y-dimension
axes[1].plot(range(len(error)), error[:, 1], label="Vector y", color="blue")
axes[1].set_xlabel("Index")
axes[1].set_ylabel("Y-dimension")
axes[1].legend()

# Plotting Z-dimension
axes[2].plot(range(len(error)), vector[:, 2], label="Vector z", color="blue")
axes[2].set_xlabel("Index")
axes[2].set_ylabel("Z-dimension")
axes[2].legend()
plt.tight_layout()
plt.show()
