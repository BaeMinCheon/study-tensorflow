import tensorflow as tf
tf.set_random_seed(777)
import matplotlib.pyplot as plt

trainX = [1, 2, 3]
trainY = [1, 2, 3]

inputX = tf.placeholder(tf.float32)
inputY = tf.placeholder(tf.float32)
weight = tf.Variable(5.0)

hypothesis = weight * inputX;
cost = tf.reduce_mean(tf.square(hypothesis - inputY))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

gradient = tf.reduce_mean((weight * inputX - inputY) * inputX) * 2
gvs = optimizer.compute_gradients(cost, [weight])
applyGradients = optimizer.apply_gradients(gvs)

weightHistory = []
costHistory = []

sess01 = tf.Session()
sess01.run(tf.global_variables_initializer())
for step in range(100):
    sess01.run(applyGradients, feed_dict={inputX: trainX, inputY: trainY})
    currentWeight = sess01.run(weight)
    currentCost = sess01.run(cost, feed_dict={inputX: trainX, inputY: trainY})
    weightHistory.append(currentWeight)
    costHistory.append(currentCost)
    print(step, currentWeight, currentCost)
    
plt.plot(weightHistory, costHistory)
plt.show()