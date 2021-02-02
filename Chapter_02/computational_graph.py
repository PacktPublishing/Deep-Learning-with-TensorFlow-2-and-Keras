import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

in_a = tf.placeholder(dtype=tf.float32, shape=(2))

def model(x):
  with tf.variable_scope("matmul"):
    W = tf.get_variable("W", initializer=tf.ones(shape=(2,2)))
    b = tf.get_variable("b", initializer=tf.zeros(shape=(2)))
    return x * W + b

out_a = model(in_a)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outs = sess.run([out_a],
                feed_dict={in_a: [1, 0]})
  writer = tf.summary.FileWriter("./logs/example", sess.graph)