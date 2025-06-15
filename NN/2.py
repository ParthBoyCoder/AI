import numpy as np

y=4

a=y
b=y
c=y

# Sigmoid and its derivative (just in case)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return sigmoid(x) * (1 - sigmoid(x))

# ReLU and its derivative
def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

# Generate dataset
def generate_data(num_samples=1000):
    X = np.random.randint(0, 11, size=(num_samples, 2))  # Two numbers 0–10
    y = np.sum(X, axis=1, keepdims=True)  # Sum as label
    return X, y

# Initialize weights
input_size = 2
hidden_size = 5 * c
output_size = 1

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Hyperparameters
epochs = 2000 * a
learning_rate = 0.001 * b

# Training data
X, y = generate_data(1000)

# Training loop
for epoch in range(epochs):
    # Forward pass
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    y_pred = z2  # Linear output

    # Loss
    loss = np.mean((y_pred - y) ** 2)

    # Backward pass
    grad_y_pred = 2 * (y_pred - y) / y.shape[0]
    grad_W2 = a1.T @ grad_y_pred
    grad_b2 = np.sum(grad_y_pred, axis=0, keepdims=True)

    grad_a1 = grad_y_pred @ W2.T
    grad_z1 = grad_a1 * relu_deriv(z1)
    grad_W1 = X.T @ grad_z1
    grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

    # Update weights
    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1
    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Test
def predict_sum(a, b):
    input_data = np.array([[a, b]])
    hidden = relu(input_data @ W1 + b1)
    output = hidden @ W2 + b2
    return output[0][0]

# 🔥 Try it out
print("\nTesting...")
for a, b in [(int(input("Enter first number: ")), int(input("Enter second number: ")))]:
    pred = predict_sum(a, b)
    print(f"{a} + {b} ≈ {pred:.2f} (actual: {a + b})")
