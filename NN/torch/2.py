import torch
import torch.nn as nn

# Map characters to numbers (a=0, b=1, ..., z=25)
char_to_num = lambda ch: ord(ch) - ord('a')
num_to_char = lambda num: chr(int(round(num)) + ord('a'))

# Training data: 'a' → 'b', 'b' → 'c', etc.
X = torch.tensor([[char_to_num(c)] for c in ['a', 'b', 'c', 'd']], dtype=torch.float32)
Y = torch.tensor([[char_to_num(c)] for c in ['b', 'c', 'd', 'e']], dtype=torch.float32)

# Model
model = nn.Linear(1, 1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    y_pred = model(X)
    loss = criterion(y_pred, Y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Test: Predict the next letter after 'e'
with torch.no_grad():
    test_input = torch.tensor([[char_to_num(input())]], dtype=torch.float32)
    prediction = model(test_input)
    predicted_char = num_to_char(prediction.item())
    print(f"Prediction  → '{predicted_char}' ({prediction.item():.2f})")
