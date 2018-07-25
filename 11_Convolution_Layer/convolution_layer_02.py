import tensorflow as tf
from Model import Model
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learningRate = 0.001
totalEpoch = 15
batchSize = 100

sess = tf.Session()
models = []
numberOfModels = 2
for m in range(numberOfModels):
    models.append(Model(sess, "model" + str(m), 0.001))
sess.run(tf.global_variables_initializer())

for epoch in range(totalEpoch):
    avgCost = np.zeros(len(models))
    totalBatch = int(mnist.train.num_examples / batchSize)

    for i in range(totalBatch):
        batchX, batchY = mnist.train.next_batch(batchSize)

        for index, model in enumerate(models):
            costValue, trainValue = model.processTrain(batchX, batchY)
            avgCost[index] += costValue / totalBatch

    print("epoch : {} \t cost : {}".format(epoch, avgCost))

testSize = len(mnist.test.labels)
prediction = np.zeros([testSize, 10])
for index, model in enumerate(models):
    print("#{} \t accu : {:.9}".format(index,
    model.getAccuracy(mnist.test.images, mnist.test.labels)))
    prediction += model.getPredict(mnist.test.images)

ensemblePrediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(mnist.test.labels, 1))
ensembleAccuracy = tf.reduce_mean(tf.cast(ensemblePrediction, tf.float32))
print("ensemble accu : {}".format(sess.run(ensembleAccuracy)))

print("DONE")