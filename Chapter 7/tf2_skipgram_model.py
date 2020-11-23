import tensorflow as tf

class SkipgramModel(tf.keras.Model):
    def __init__(self, vocab_sz, embed_sz, **kwargs):
        super(SkipgramModel, self).__init__(**kwargs)
        embedding = tf.keras.layers.Embedding(
            input_dim=vocab_sz,
            output_dim=embed_sz,
            embeddings_initializer="glorot_uniform",
            input_length=1
        )
        self.word_model = tf.keras.Sequential([
            embedding,
            tf.keras.layers.Flatten()
        ])
        self.context_model = tf.keras.Sequential([
            embedding,
            tf.keras.layers.Flatten()
        ])
        self.merge = tf.keras.layers.Dot(axes=1)
        self.dense = tf.keras.layers.Dense(1,
                kernel_initializer="glorot_uniform",
                activation="sigmoid"
        )

    def call(self, input):
        word, context = input
        word_emb = self.word_model(word)
        context_emb = self.context_model(context)
        x = self.merge([word_emb, context_emb])
        x = self.dense(x)
        return x


VOCAB_SIZE = 5000
EMBED_SIZE = 300

model = SkipgramModel(VOCAB_SIZE, EMBED_SIZE)
model.build(input_shape=[(None, VOCAB_SIZE), (None, VOCAB_SIZE)])
model.compile(optimizer=tf.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"])

model.summary()

# train the model here

# retrieve embeddings from trained model
word_model = model.layers[0]
word_emb_layer = word_model.layers[0]
emb_weights = None
for weight in word_emb_layer.weights:
    if weight.name == "embedding/embeddings:0":
        emb_weights = weight.numpy()
print(emb_weights, emb_weights.shape)

