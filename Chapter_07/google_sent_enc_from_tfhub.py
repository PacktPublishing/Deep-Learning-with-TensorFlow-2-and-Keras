import tensorflow as tf
import tensorflow_hub as hub

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
tf.compat.v1.disable_eager_execution()

model = hub.Module(module_url)
embeddings = model([
    "i like green eggs and ham",
    "would you eat them in a box"
])
with tf.compat.v1.Session() as sess:
    sess.run([
        tf.compat.v1.global_variables_initializer(),
        tf.compat.v1.tables_initializer()
    ])
    embeddings_value = sess.run(embeddings)

print(embeddings_value.shape)
