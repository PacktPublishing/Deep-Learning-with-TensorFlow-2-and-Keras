import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.Sequential([
	keras.layers.Dense(12, input_dim=8, name='dense_layer', 
		kernel_initializer='random_uniform'),
])