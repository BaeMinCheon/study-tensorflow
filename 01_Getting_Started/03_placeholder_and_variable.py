import tensorflow as tf

sess = tf.Session()

input = tf.placeholder(tf.float32, [1, 3])
weight = tf.Variable([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
bias = tf.Variable([1.0, 1.0])
output = tf.matmul(input, weight) + bias
data = [[1, 2, 3]]
sess.run(tf.global_variables_initializer())
print("input * weight + bias \n -> {}".format(sess.run(output, feed_dict={input: data})))

input = tf.placeholder(tf.float32, [None, 3])
weight = tf.Variable([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
bias = tf.Variable([1.0, 1.0])
output = tf.matmul(input, weight) + bias
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
sess.run(tf.global_variables_initializer())
print("input * weight + bias \n -> {}".format(sess.run(output, feed_dict={input: data})))