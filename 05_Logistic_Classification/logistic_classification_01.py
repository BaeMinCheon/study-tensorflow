import tensorflow as tf
tf.set_random_seed(777)

import os
dirPath = os.path.dirname(os.path.abspath(__file__))
filePath = dirPath + '/data.csv'

trainX = [
    [1, 2],
    [2, 3],
    [3, 1],
    [4, 3],
    [5, 3],
    [6, 2]
]
trainY = [
    [0],
    [0],
    [0],
    [1],
    [1],
    [1]
]

inputX = tf.placeholder(tf.float32, shape=[None, 2])
inputY = tf.placeholder(tf.float32, shape=[None, 1])

weight = tf.Variable(tf.random_normal([2, 1]), name='weight')
bias = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(inputX, weight) + bias)
cost = tf.reduce_mean(-inputY * tf.log(hypothesis) -(1 - inputY) * tf.log(1 - hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, inputY), dtype=tf.float32))

sess01 = tf.Session()
sess01.run(tf.global_variables_initializer())

for step in range(2001):
    costValue, hypoValue, trainValue = sess01.run([cost, hypothesis, train],
    feed_dict={inputX: trainX, inputY: trainY})
    if step % 200 == 0:
        print(step, costValue)

hypo, pred, accu = sess01.run([hypothesis, predicted, accuracy],
feed_dict={inputX: trainX, inputY: trainY})
print('test with train data')
print('( hypo, pred, accu ) -> ', hypo, pred, accu)

print("DONE")