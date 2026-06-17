import tensorflow as tf
import numpy as np
from tensorflow import keras


N_TRAIN_EXAMPLES = 1024*1024
N_FEATURES = 10
SIZE_BATCHES = 256

# 10 random floats in the half-open interval [0.0, 1.0).
x = np.random.random((N_TRAIN_EXAMPLES, N_FEATURES))
y = np.random.randint(2, size=(N_TRAIN_EXAMPLES, 1))
x = tf.dtypes.cast(x, tf.float32)
print (x)
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(buffer_size=N_TRAIN_EXAMPLES).batch(SIZE_BATCHES)


# this is the distribution strategy
distribution = tf.distribute.MirroredStrategy()

# this piece of code is distributed to multiple GPUs
with distribution.scope():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(N_FEATURES,)))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
  optimizer = tf.keras.optimizers.SGD(0.2)
  model.compile(loss='binary_crossentropy', optimizer=optimizer)

model.summary()

# Optmize in the usual way but in reality you are using GPUs.
model.fit(dataset, epochs=5, steps_per_epoch=100)

