import torch
import torch.nn as nn

char_to_num = lambda ch: ord(ch) - ord('a')
num_to_char = lambda num: chr(int(round(num)) + ord('a'))

X = torch.tensor([[char_to_num(c)] for c in ['a', 'b', 'c', 'd']], dtype=torch.float32)
Y = torch.tensor([[char_to_num(c)] for c in ['c', 'd', 'e', 'f']], dtype=torch.float32)

model = nn.Linear(1, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(X)
    loss = criterion(y_pred, Y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

with torch.no_grad():
    test_input = torch.tensor([[char_to_num(input())]], dtype=torch.float32)
    prediction = model(test_input)
    predicted_char = num_to_char(prediction.item())
    print(f"Prediction  → '{predicted_char}' ({prediction.item():.2f})")
