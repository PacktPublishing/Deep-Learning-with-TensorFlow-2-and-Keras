import numpy as np
import tensorflow as tf

batch_size = 16
num_timesteps = 10
embedding_dim = 128
hidden_dim = 256

inputs = tf.keras.Input(shape=(num_timesteps, embedding_dim))

peephole_lstm_cell = tf.keras.experimental.PeepholeLSTMCell(hidden_dim)
rnn_layer = tf.keras.layers.RNN(peephole_lstm_cell)

outputs = rnn_layer(inputs)

model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.summary()

input = np.random.uniform(size=(batch_size, num_timesteps, embedding_dim))
output = model.predict(input)
print("input shape:", input.shape, "output shape:", output.shape)
