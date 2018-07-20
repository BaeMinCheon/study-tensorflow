import tensorflow as tf
tf.set_random_seed(777)

import os
dirPath = os.path.dirname(os.path.abspath(__file__))
filePath = dirPath + '/data.csv'

trainX = [
    [1, 2, 1, 1],
    [2, 1, 3, 2],
    [3, 1, 3, 4],
    [4, 1, 5, 5],
    [1, 7, 5, 5],
    [1, 2, 5, 6],
    [1, 6, 6, 6],
    [1, 7, 7, 7]
]
trainY = [
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0]
]

numberOfClasses = 3
inputX = tf.placeholder(tf.float32, shape=[None, 4])
inputY = tf.placeholder(tf.float32, shape=[None, numberOfClasses])

weight = tf.Variable(tf.random_normal([4, numberOfClasses]), name='weight')
bias = tf.Variable(tf.random_normal([numberOfClasses]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(inputX, weight) + bias)
cost = tf.reduce_mean(tf.reduce_sum(-inputY * tf.log(hypothesis), axis=1))
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