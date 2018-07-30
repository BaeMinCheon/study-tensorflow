import tensorflow as tf

sess = tf.Session()

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
print("2 + 3 \n -> {}".format(sess.run(x)))

a = tf.constant([2, 2])
b = tf.constant([[0, 1], [2, 3]])
x = tf.multiply(a, b)
print("[2, 2] * [[0, 1], [2, 3]] \n -> {}".format(sess.run(x)))

a = tf.constant([[2, 2]])
b = tf.constant([[0, 1], [2, 3]])
x = tf.matmul(a, b)
print("[ 2 2 ] * [ 0 1 ; 2 3 ] \n -> {}".format(sess.run(x)))