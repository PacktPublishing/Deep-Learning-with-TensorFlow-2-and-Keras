import tensorflow as tf

x = tf.constant(4.0)
with tf.GradientTape(persistent=True) as g:
  g.watch(x)
  y = x * x
  z = y * y
dz_dx = g.gradient(z, x)  # 108.0 (4*x^3 at x = 4)
dy_dx = g.gradient(y, x)  # 6.0
print (dz_dx)
print (dy_dx)
del g  # Drop the reference to the tape