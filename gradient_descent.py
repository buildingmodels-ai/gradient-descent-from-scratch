import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic linear data: y = 2x + 1 + noise
np.random.seed(42)

X = np.linspace(0, 10, 100)
true_w = 2
true_b = 1
noise = np.random.normal(0, 1, 100)

y = true_w * X + true_b + noise

# Initialize parameters
w = 0.0
b = 0.0

learning_rate = 0.01
n = len(X)

loss_history = []

# Gradient Descent
for epoch in range(100):

    # Predictions
    y_pred = w * X + b

    # Compute loss (Mean Squared Error)
    loss = np.mean((y_pred - y) ** 2)
    loss_history.append(loss)

    # Compute gradients
    dw = (2/n) * np.sum((y_pred - y) * X)
    db = (2/n) * np.sum(y_pred - y)

    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db

# Results
print("Learned parameters:")
print(f"w = {w:.4f}")
print(f"b = {b:.4f}")

# Plot loss curve
plt.plot(loss_history)
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()
