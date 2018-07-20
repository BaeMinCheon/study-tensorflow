import tensorflow as tf
tf.set_random_seed(777)
import random
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

numberOfClasses = 10
inputX = tf.placeholder(tf.float32, [None, 784])
inputY = tf.placeholder(tf.float32, [None, numberOfClasses])

weight = tf.Variable(tf.random_normal([784, numberOfClasses]))
bias = tf.Variable(tf.random_normal([numberOfClasses]))

hypothesis = tf.nn.softmax(tf.matmul(inputX, weight) + bias)
cost = tf.reduce_mean(tf.reduce_sum(-inputY * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

isCorrect = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(inputY, 1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

epochSize = 15
batchSize = 100

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(epochSize):
    avgCost = 0
    totalBatch = int(mnist.train.num_examples / batchSize)

    for i in range(totalBatch):
        batchX, batchY = mnist.train.next_batch(batchSize)
        costValue, trainValue = sess.run([cost, train],
        feed_dict={inputX: batchX, inputY: batchY})
        avgCost += costValue / totalBatch

    print("epoch : #{:4} \t avgCost : {:.9f}".format(epoch, avgCost))

print("accuracy : {:.9f}".format(accuracy.eval(session=sess,
feed_dict={inputX: mnist.test.images, inputY: mnist.test.labels})))

rand = random.randint(0, mnist.test.num_examples - 1)
print("label : ", sess.run(tf.argmax(mnist.test.labels[rand : rand+1], 1)))
print("prediction : ", sess.run(tf.argmax(hypothesis, 1),
feed_dict={inputX: mnist.test.images[rand : rand+1]}))

plt.imshow(mnist.test.images[rand : rand+1].reshape(28, 28),
cmap='Greys', interpolation='nearest')
plt.show()