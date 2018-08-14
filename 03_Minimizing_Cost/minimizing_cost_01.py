import tensorflow as tf
tf.set_random_seed(777)
import matplotlib.pyplot as plt

trainX = [1, 2, 3]
trainY = [1, 2, 3]

inputX = tf.placeholder(tf.float32)
inputY = tf.placeholder(tf.float32)
weight = tf.Variable(tf.random_normal([1]), name='weight')

hypothesis = weight * inputX
cost = tf.reduce_mean(tf.square(hypothesis - inputY))

learning_rate = 0.1
gradient = tf.reduce_mean((weight * inputX - inputY) * inputX)
descent = weight - learning_rate * gradient
update = weight.assign(descent)

weightHistory = []
costHistory = []

sess01 = tf.Session()
sess01.run(tf.global_variables_initializer())
for step in range(21):
    sess01.run(update, feed_dict={inputX: trainX, inputY: trainY})
    currentWeight = sess01.run(weight)
    currentCost = sess01.run(cost, feed_dict={inputX: trainX, inputY: trainY})
    weightHistory.append(currentWeight)
    costHistory.append(currentCost)
    print(step, currentCost, currentWeight)
    
plt.plot(weightHistory, costHistory)
plt.show()