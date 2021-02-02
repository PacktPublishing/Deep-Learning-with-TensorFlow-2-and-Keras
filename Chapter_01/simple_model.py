import tensorflow as tf
W = tf.Variable(tf.ones(shape=(2,2)), name="W")
b = tf.Variable(tf.zeros(shape=(2)), name="b")

@tf.function
def model(x):
  return W * x + b
out_a = model([1,0])

print(out_a) 
