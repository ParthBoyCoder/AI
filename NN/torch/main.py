import torch
import torch.nn as nn

# Training data (y = 2x)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Define a model: 1 input -> 1 output
model = nn.Linear(in_features=1, out_features=1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, Y)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Test the model
with torch.no_grad():
    test_input = torch.tensor([[5.0]])
    prediction = model(test_input)
    print(f"Prediction for 5.0: {prediction.item():.4f}")
