import tensorflow as tf
import numpy as np

# Training data
x = np.array([[0.], [1.], [2.], [3.], [4.]], dtype=float)
y = np.array([[0.], [3.], [6.], [9.], [12.]], dtype=float)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train
model.fit(x, y, epochs=400, verbose=0)

# Predict
print("Prediction for 10:", model.predict(np.array([[10.]])))
