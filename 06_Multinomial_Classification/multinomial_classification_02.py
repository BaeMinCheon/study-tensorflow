import tensorflow as tf
tf.set_random_seed(777)
import os
dirPath = os.path.dirname(os.path.abspath(__file__))
filePath = dirPath + '/data.csv'
import numpy as np
train_data = np.loadtxt(filePath, delimiter=',', dtype=np.float32)

trainX = train_data[:, 0:-1]
trainY = train_data[:, [-1]]
print(trainX.shape, trainY.shape)

numberOfClasses = 7
inputX = tf.placeholder(tf.float32, shape=[None, 16])
inputY = tf.placeholder(tf.int32, shape=[None, 1])
oneHotY = tf.one_hot(inputY, numberOfClasses)
print("oneHotY, after one_hot() : ", oneHotY)
oneHotY = tf.reshape(oneHotY, [-1, numberOfClasses])
print("oneHotY, after reshape() : ", oneHotY)

weight = tf.Variable(tf.random_normal([16, numberOfClasses]), name='weight')
bias = tf.Variable(tf.random_normal([numberOfClasses]), name='bias')
logit = tf.matmul(inputX, weight) + bias

hypothesis = tf.nn.softmax(logit)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=oneHotY))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

prediction = tf.argmax(hypothesis, 1)
isCorrect = tf.equal(prediction, tf.argmax(oneHotY, 1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

sess01 = tf.Session()
sess01.run(tf.global_variables_initializer())

for step in range(2001):
    sess01.run(train, feed_dict={inputX: trainX, inputY: trainY})
    if step % 200 == 0:
        costValue, accuracyValue = sess01.run([cost, accuracy],
        feed_dict={inputX: trainX, inputY: trainY})
        print("step #{:4} \t cost {:.3f} \t accuracy {:.2}".format(step, costValue, accuracyValue))

test01 = sess01.run(prediction, feed_dict={inputX: trainX})
for pred, real in zip(test01, trainY.flatten()):
    print("is equal ? [{}] \t pred {} \t real {}".format(pred == int(real), pred, int(real)))

print("DONE")