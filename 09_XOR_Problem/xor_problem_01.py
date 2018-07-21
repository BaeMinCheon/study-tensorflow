import tensorflow as tf
tf.set_random_seed(777)
import numpy as np
import os
dirPath = os.path.dirname(os.path.abspath(__file__))
filePath = dirPath + '/graph_xor'

learning_rate = float(input("input learning rate : "))

trainX = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
trainX = np.array(trainX, dtype=np.float32)
trainY = [[0],
          [1],
          [1],
          [0]]
trainY = np.array(trainY, dtype=np.float32)

inputX = tf.placeholder(tf.float32, [None, 2], name='inputX')
inputY = tf.placeholder(tf.float32, [None, 1], name='inputY')

with tf.name_scope("layer01"):
    weight01 = tf.Variable(tf.random_normal([2, 2]), name='weight01')
    bias01 = tf.Variable(tf.random_normal([2]), name='bias01')
    layer01 = tf.sigmoid(tf.matmul(inputX, weight01) + bias01)
    
    w1Hist = tf.summary.histogram("WH01", weight01)
    b1Hist = tf.summary.histogram("BH01", bias01)
    l1Hist = tf.summary.histogram("LH01", layer01)

with tf.name_scope("layer2"):
    weight02 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
    bias02 = tf.Variable(tf.random_normal([1]), name='bias2')
    layer02 = tf.sigmoid(tf.matmul(layer01, weight02) + bias02)
    
    w2Hist = tf.summary.histogram("WH02", weight02)
    b2Hist = tf.summary.histogram("BH02", bias02)
    l2Hist = tf.summary.histogram("LH02", layer02)

with tf.name_scope("cost"):
    cost = tf.reduce_mean(-inputY * tf.log(layer02) - (1 - inputY) *
                           tf.log(1 - layer02))
    costSum = tf.summary.scalar("cost", cost)

with tf.name_scope("train"):
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.cast(layer02 > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, inputY), dtype=tf.float32))
accuracySum = tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
    mergeSum = tf.summary.merge_all()
    writer = tf.summary.FileWriter(filePath, sess.graph)
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        mergeValue, trainValue = sess.run([mergeSum, train],
        feed_dict={inputX: trainX, inputY: trainY})
        writer.add_summary(mergeValue, global_step=step)

        if step % 500 == 0:
            print(step, sess.run(cost,
            feed_dict={inputX: trainX, inputY: trainY}), sess.run([weight01, weight02]))

    hypoValue, predValue, accuValue = sess.run([layer02, predicted, accuracy],
                       feed_dict={inputX: trainX, inputY: trainY})
    print("\nHypothesis: ", hypoValue, "\nCorrect: ", predValue, "\nAccuracy: ", accuValue)

print("DONE")

# after train, type in terminal "tensorboard --logdir=[dirPath]/graph_xor"
# if tensorboard starts successfully, it would tell you a link starting "http..."
# open up any web browser and paste the link, now good to go