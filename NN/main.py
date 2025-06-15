import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize inputs to 0–1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to one-hot
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate on test data
loss, acc = model.evaluate(x_test, y_test)
print(f"\n:dart: Final Test Accuracy: {acc * 100:.2f}%")

# Predict and compare
preds = model.predict(x_test[:5])
for i in range(5):
    predicted = np.argmax(preds[i])
    actual = np.argmax(y_test[i])
    print(f"Image {i+1}: Predicted = {predicted}, Actual = {actual}")
