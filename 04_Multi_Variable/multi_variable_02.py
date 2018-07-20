import tensorflow as tf
tf.set_random_seed(777)
import numpy as np
import os

dirPath = os.path.dirname(os.path.abspath(__file__))
filePath = dirPath + '/data.csv'

train_data = np.loadtxt(filePath, delimiter=',', dtype=np.float32)
trainX = train_data[:, 0:-1]
trainY = train_data[:, [-1]]
print(trainX.shape, trainX, len(trainX))
print(trainY.shape, trainY, len(trainY))

inputX = tf.placeholder(tf.float32, shape=[None, 3])
inputY = tf.placeholder(tf.float32, shape=[None, 1])

weight = tf.Variable(tf.random_normal([3, 1]), name='weight')
bias = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(inputX, weight) + bias
print("hypo -> ", hypothesis)
cost = tf.reduce_mean(tf.square(hypothesis - inputY))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess01 = tf.Session()
sess01.run(tf.global_variables_initializer())
for step in range(2001):
    costValue, hypoValue, trainValue = sess01.run([cost, hypothesis, train],
    feed_dict={inputX: trainX, inputY: trainY})
    if step % 100 == 0:
        print()
        print(step, "( cost, hypo ) -> ", costValue, hypoValue)
        print()

test01 = [[100, 70, 101]]
print("test #1 with", test01)
print(sess01.run(hypothesis, feed_dict={inputX: test01}))

test02 = [[60, 70, 110], [90, 100, 80]]
print("test #2 with", test02)
print(sess01.run(hypothesis, feed_dict={inputX: test02}))

print("DONE")