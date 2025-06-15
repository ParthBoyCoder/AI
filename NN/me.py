import numpy as np

x=100

a=x
b=x
c=x

# XOR Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and biases randomly
np.random.seed(42)
input_layer_size = 2
hidden_layer_size = 2*a
output_layer_size = 1

# Weights
W1 = np.random.uniform(size=(input_layer_size, hidden_layer_size))
W2 = np.random.uniform(size=(hidden_layer_size, output_layer_size))

# Biases
b1 = np.random.uniform(size=(1, hidden_layer_size))
b2 = np.random.uniform(size=(1, output_layer_size))

# Training the network
epochs = 10000*b
learning_rate = 0.1*c

for epoch in range(epochs):
    # Forward Propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    predicted_output = sigmoid(final_input)

    # Error calculation
    error = y - predicted_output
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Error: {np.mean(np.abs(error)):.4f}")

    # Backpropagation
    d_output = error * sigmoid_derivative(predicted_output)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)

    # Update weights and biases
    W2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Final predictions
print("\nFinal Predictions after training:")
for i in range(4):
    print(f"{X[i]} => {predicted_output[i][0]:.4f} (Expected: {y[i][0]})")