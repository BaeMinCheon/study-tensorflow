import tensorflow as tf
tf.set_random_seed(777)

import os
dirPath = os.path.dirname(os.path.abspath(__file__))
filePath = dirPath + '/data.csv'

fileQueue = tf.train.string_input_producer([filePath], shuffle=False, name='fileQueue')
reader = tf.TextLineReader()
key, value = reader.read(fileQueue)

recordFormat = [[0.], [0.], [0.], [0.]]
train_data = tf.decode_csv(value, record_defaults=recordFormat)
trainX_batch, trainY_batch = tf.train.batch([train_data[0:-1], train_data[-1:]], batch_size=10)

inputX = tf.placeholder(tf.float32, shape=[None, 3])
inputY = tf.placeholder(tf.float32, shape=[None, 1])

weight = tf.Variable(tf.random_normal([3, 1]), name='weight')
bias = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(inputX, weight) + bias
cost = tf.reduce_mean(tf.square(hypothesis - inputY))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess01 = tf.Session()
sess01.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess01, coord=coord)
for step in range(2001):
    batchX, batchY = sess01.run([trainX_batch, trainY_batch])
    costValue, hypoValue, trainValue = sess01.run([cost, hypothesis, train],
    feed_dict={inputX: batchX, inputY: batchY})
coord.request_stop()
coord.join(threads)

test01 = [[100, 70, 101]]
print("test #1 with", test01)
print(sess01.run(hypothesis, feed_dict={inputX: test01}))

test02 = [[60, 70, 110], [90, 100, 80]]
print("test #2 with", test02)
print(sess01.run(hypothesis, feed_dict={inputX: test02}))

print("DONE")