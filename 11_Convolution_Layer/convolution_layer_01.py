import tensorflow as tf
tf.set_random_seed(777)
import random

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learningRate = 0.001
totalEpoch = 15
batchSize = 100
keepProb = tf.placeholder(tf.float32)

inputX = tf.placeholder(tf.float32, [None, 784])
inputImg = tf.reshape(inputX, [-1, 28, 28, 1])
inputY = tf.placeholder(tf.float32, [None, 10])

weight01 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
layer01 = tf.nn.conv2d(inputImg, weight01, strides=[1, 1, 1, 1], padding='SAME')
layer01 = tf.nn.relu(layer01)
layer01 = tf.nn.max_pool(layer01, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
layer01 = tf.nn.dropout(layer01, keep_prob=keepProb)

weight02 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
layer02 = tf.nn.conv2d(layer01, weight02, strides=[1, 1, 1, 1], padding='SAME')
layer02 = tf.nn.relu(layer02)
layer02 = tf.nn.max_pool(layer02, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
layer02 = tf.nn.dropout(layer02, keep_prob=keepProb)

weight03 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
layer03 = tf.nn.conv2d(layer02, weight03, strides=[1, 1, 1, 1], padding='SAME')
layer03 = tf.nn.relu(layer03)
layer03 = tf.nn.max_pool(layer03, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
layer03 = tf.reshape(layer03, [-1, 128 * 4 * 4])

weight04 = tf.get_variable("weight04", shape=[128 * 4 * 4, 625],
initializer=tf.contrib.layers.xavier_initializer())
bias04 = tf.Variable(tf.random_normal([625]))
layer04 = tf.nn.relu(tf.matmul(layer03, weight04) + bias04)
layer04 = tf.nn.dropout(layer04, keep_prob=keepProb)

weight05 = tf.get_variable("weight05", shape=[625, 10],
initializer=tf.contrib.layers.xavier_initializer())
bias05 = tf.Variable(tf.random_normal([10]))
layer05 = tf.matmul(layer04, weight05) + bias05

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer05, labels=inputY))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(totalEpoch):
    avgCost = 0
    totalBatch = int(mnist.train.num_examples / batchSize)

    for i in range(totalBatch):
        batchX, batchY = mnist.train.next_batch(batchSize)
        costValue, trainValue = sess.run([cost, train],
        feed_dict={inputX: batchX, inputY: batchY, keepProb: 0.7})
        avgCost += costValue

    avgCost /= totalBatch
    print("epochNum : {:4} \t avgCost : {:.9}".format(epoch, avgCost))

prediction = tf.equal(tf.argmax(layer05, 1), tf.argmax(inputY, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print("accu : {}".format(sess.run(accuracy,
feed_dict={inputX: mnist.test.images, inputY: mnist.test.labels, keepProb: 1})))

print("DONE")