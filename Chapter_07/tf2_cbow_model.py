import tensorflow as tf

class CBOWModel(tf.keras.Model):
    def __init__(self, vocab_sz, emb_sz, window_sz, **kwargs):
        super(CBOWModel, self).__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_sz,
            output_dim=emb_sz,
            embeddings_initializer="glorot_uniform",
            input_length=window_sz*2
        )
        self.dense = tf.keras.layers.Dense(
            vocab_sz,
            kernel_initializer="glorot_uniform",
            activation="softmax"
        )

    def call(self, x):
        x = self.embedding(x)
        x = tf.reduce_mean(x, axis=1)
        x = self.dense(x)
        return x


VOCAB_SIZE = 5000
EMBED_SIZE = 300
WINDOW_SIZE = 1  # 3 word window, 1 on left, 1 on right

model = CBOWModel(VOCAB_SIZE, EMBED_SIZE, WINDOW_SIZE)
model.build(input_shape=(None, VOCAB_SIZE))
model.compile(optimizer=tf.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"])

model.summary()

# train the model here

# retrieve embeddings from trained model
emb_layer = [layer for layer in model.layers 
    if layer.name.startswith("embedding")][0]
emb_weight = [weight.numpy() for weight in emb_layer.weights
    if weight.name.endswith("/embeddings:0")][0]
print(emb_weight, emb_weight.shape)


