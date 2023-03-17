import tensorflow as tf
import tensorflow_datasets as tfds

BUFFER_SIZE = 10000
BATCH_SIZE = 64

def input_fn(mode):
  datasets, info = tfds.load(name='mnist',
                                with_info=True,
                                as_supervised=True)
  mnist_dataset = (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else
                   datasets['test'])

  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  return mnist_dataset.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test = input_fn('test')
train = input_fn(tf.estimator.ModeKeys.TRAIN)

print(test)
print(train)