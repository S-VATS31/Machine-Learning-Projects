import torch
import math
import matplotlib.pyplot as plt

def tanh(x):
    """Compute tanh(x) element-wise"""
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

def tanh_derivative(x):
    """Compute tanh(x) derivative element-wise"""
    t = tanh(x)
    return 1 - (t ** 2)

def gradient_descent(x, y, weights, biases, alpha):
    """Apply gradient descent"""

    # Forward pass (Linear transformation)
    z = torch.matmul(x, weights) + biases

    # Tanh activation (non-linear activation)
    a = tanh(z)

    # Compute loss function
    loss = torch.mean(0.5 * (a - y) ** 2)

    # Backward pass (compute derivatives)
    dL_da = a - y
    da_dz = tanh_derivative(z)
    delta = dL_da * da_dz

    # Apply chain rule
    dL_dW = torch.matmul(x.T, delta) / x.size(0)
    dL_db = delta.mean(dim=0)

    # Gradient descent update rule
    with torch.no_grad():
        weights -= alpha * dL_dW
        biases -= alpha * dL_db

    return loss.item()

def initialize_weights(weights, biases, uniform=False):
    """Initialize weights using Xavier initialization"""
    fan_in, fan_out = weights.shape

    # Apply uniform initialization
    if uniform:
        limit = math.sqrt(6 / (fan_in + fan_out))
        with torch.no_grad():
            weights.uniform_(-limit, limit)

    # Apply normal initialization
    else:
        std = math.sqrt(2 / (fan_in + fan_out))
        with torch.no_grad():
            weights.normal_(0, std)

    # Initalize biases
    if biases is not None:
        with torch.no_grad():
            biases.zero_()

    return weights, biases

# Input tensor
x = torch.randn(4, 3)

# True output tensor
y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

# Define parameters
weights = torch.empty(3, 1)
biases = torch.empty(1)

# Initialize parameters
weights, biases = initialize_weights(weights, biases, True)

# Initialize list of losses
losses = []

# Training loop
for epoch in range(1000):
    loss = gradient_descent(x, y, weights, biases, alpha=0.1)
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss = {loss:.4f}")

# Plot loss curve
plt.plot(range(len(losses)), losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over epochs")
plt.grid(True)
plt.show()
