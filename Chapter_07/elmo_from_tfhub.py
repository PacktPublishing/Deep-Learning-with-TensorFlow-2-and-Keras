import tensorflow as tf
import tensorflow_hub as hub

# module_url = "https://tfhub.dev/google/tf2-preview/elmo/2"
# embed = hub.KerasLayer(module_url)
# embeddings = embed([
#     "i like green eggs and ham",
#     "would you eat them in a box"
# ])
# print(embeddings.shape)

module_url = "https://tfhub.dev/google/elmo/2"
tf.compat.v1.disable_eager_execution()
elmo = hub.Module(module_url, trainable=False)
embeddings = elmo([
        "i like green eggs and ham",
        "would you eat them in a box"
    ], 
    signature="default",
    as_dict=True
)["elmo"]
print(embeddings.shape)
