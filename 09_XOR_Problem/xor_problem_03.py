import tensorflow as tf
tf.set_random_seed(777)
import numpy as np

learningRate = 0.1

trainX = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
trainX = np.array(trainX, dtype=np.float32)
trainY = [[0],
          [1],
          [1],
          [0]]
trainY = np.array(trainY, dtype=np.float32)

inputX = tf.placeholder(tf.float32, [None, 2])
inputY = tf.placeholder(tf.float32, [None, 1])

weight01 = tf.Variable(tf.random_normal([2, 2]), name='weight01')
bias01 = tf.Variable(tf.random_normal([2]), name='bias01')
layer01 = tf.sigmoid(tf.matmul(inputX, weight01) + bias01)

weight02 = tf.Variable(tf.random_normal([2, 1]), name='weight02')
bias02 = tf.Variable(tf.random_normal([1]), name='bias02')
layer02 = tf.sigmoid(tf.matmul(layer01, weight02) + bias02)

cost = tf.reduce_mean(- inputY * tf.log(layer02) - (1 - inputY) * tf.log(1 - layer02))

# Network
#              mul01  add01      layer01  mul02  add02      layer02
# inputX -> (*) -> (+) -> (sigmoid) -> (*) -> (+) -> (sigmoid) -> (loss)
#            ^      ^                   ^      ^
#            |      |                   |      |
#      weight01     bias01        weight02     bias02

# loss function
devLayer02 = (layer02 - inputY) / (layer02 * (1.0 - layer02) + 1e-7)

# sigmoid f(x) = 1 / ( 1 + e^(-x))
# dev of sigmoid = f(x) * (1 - f(x))
d_sigma2 = layer02 * (1 - layer02)
d_a2 = devLayer02 * d_sigma2
d_p2 = d_a2
d_bias02 = d_a2
d_weight02 = tf.matmul(tf.transpose(layer01), d_p2)

d_bias02_mean = tf.reduce_mean(d_bias02, axis=[0])
d_weight02_mean = d_weight02 / tf.cast(tf.shape(layer01)[0], dtype=tf.float32)

d_layer01 = tf.matmul(d_p2, tf.transpose(weight02))
d_sigma1 = layer01 * (1 - layer01)
d_a1 = d_layer01 * d_sigma1
d_bias01 = d_a1
d_p1 = d_a1
d_weight01 = tf.matmul(tf.transpose(inputX), d_a1)

d_weight01_mean = d_weight01 / tf.cast(tf.shape(inputX)[0], dtype=tf.float32)
d_bias01_mean = tf.reduce_mean(d_bias01, axis=[0])

step = [
  tf.assign(weight02, weight02 - learningRate * d_weight02_mean),
  tf.assign(bias02, bias02 - learningRate * d_bias02_mean),
  tf.assign(weight01, weight01 - learningRate * d_weight01_mean),
  tf.assign(bias01, bias01 - learningRate * d_bias01_mean)
]

predicted = tf.cast(layer02 > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, inputY), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("shape", sess.run(tf.shape(inputX)[0], feed_dict={inputX: trainX}))

    for i in range(10001):
        sess.run([step, cost], feed_dict={inputX: trainX, inputY: trainY})
        if i % 1000 == 0:
            print(i, sess.run([cost, d_weight01], feed_dict={
                  inputX: trainX, inputY: trainY}), sess.run([weight01, weight02]))

    hypo, pred, accu = sess.run([layer02, predicted, accuracy],
                       feed_dict={inputX: trainX, inputY: trainY})
    print("\n hypo: ", hypo, "\n pred: ", pred, "\n accu: ", accu)

print("DONE")