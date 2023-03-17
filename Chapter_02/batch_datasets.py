import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

builder = tfds.builder('imdb_reviews')
builder.download_and_prepare()


datasets, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

train_dataset = datasets['train']
train_dataset = train_dataset.batch(5).shuffle(50).take(2)

for data in train_dataset:
    print(data)
