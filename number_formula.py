import numpy as np
import tensorflow as tf;

keras = tf.keras
# Create a tensorflow model that is a single layer with keras,
# units=1 specifies that there will be one output neuron in this layer. 
# This means that the layer will output a single scalar value, which is a real number.
# input_shape=[1] specifies the shape of the input to this layer. In this case, 
# the input is a one-dimensional array with one element. This means that the input to this 
# layer is a single scalar value, which is a real number
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# Guessing the data
# optimizer='sgd' generate another guess
# loss='mean_square_error' determines how good or bad the guess
model.compile(optimizer='sgd', loss='mean_squared_error')
# Model for formula:
# y = 2x-1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
# Loop 500 times for guessing
model.fit(xs, ys, epochs=500)
# This might have a discrepancy for decimals but round it up
print(model.predict([10.0, 11.0]))