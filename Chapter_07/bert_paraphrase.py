import os
import tensorflow as tf
import tensorflow_datasets
from transformers import BertTokenizer, BertForSequenceClassification, TFBertForSequenceClassification, glue_convert_examples_to_features

BATCH_SIZE = 32
FINE_TUNED_MODEL_DIR = "./data/"

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")

# load data
data, info = tensorflow_datasets.load("glue/mrpc", with_info=True)
num_train = info.splits["train"].num_examples
num_valid = info.splits["validation"].num_examples

# Prepare dataset for GLUE as a tf.data.Dataset instance
Xtrain = glue_convert_examples_to_features(data["train"], tokenizer, 128, "mrpc")
Xtrain = Xtrain.shuffle(128).batch(32).repeat(-1)
Xvalid = glue_convert_examples_to_features(data["validation"], tokenizer, 128, "mrpc")
Xvalid = Xvalid.batch(32)

opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
model.compile(optimizer=opt, loss=loss, metrics=[metric])

# Train and evaluate using tf.keras.Model.fit()
train_steps = num_train // 32
valid_steps = num_valid // 32

history = model.fit(Xtrain, epochs=2, steps_per_epoch=train_steps,
    validation_data=Xvalid, validation_steps=valid_steps)

model.save_pretrained(FINE_TUNED_MODEL_DIR)

# load saved model
saved_model = BertForSequenceClassification.from_pretrained(
    FINE_TUNED_MODEL_DIR, from_tf=True)

# predict sentence paraphrase
sentence_0 = "At least 12 people were killed in the battle last week."
sentence_1 = "At least 12 people lost their lives in last weeks fighting."
sentence_2 = "The fires burnt down the houses on the street."

inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, return_tensors="pt")
inputs_2 = tokenizer.encode_plus(sentence_0, sentence_2, return_tensors="pt")

pred_1 = saved_model(**inputs_1)[0].argmax().item()
pred_2 = saved_model(**inputs_2)[0].argmax().item()

def print_result(id1, id2, pred):
    if pred == 1:
        print("sentence_1 is a paraphrase of sentence_0")
    else:
        print("sentence_1 is not a paraphrase of sentence_0")

print_result(0, 1, pred_1)
print_result(0, 2, pred_2)

