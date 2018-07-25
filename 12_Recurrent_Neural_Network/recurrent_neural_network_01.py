import tensorflow as tf
tf.set_random_seed(777)
import numpy as np

id2char = ['h', 'i', 'e', 'l', 'o']
# h i h e l l
inputData = [[0, 1, 0, 2, 3, 3]]
inputOneHot = [[
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0]
]]
# i h e l l o
labelData = [[1, 0, 2, 3, 3, 4]]

numberOfClasses = 5
inputDim = 5
hiddenSize = 5
batchSize = 1
sequenceLength = 6
learningRate = 0.1

inputX = tf.placeholder(tf.float32, [None, sequenceLength, inputDim])
inputY = tf.placeholder(tf.int32, [None, sequenceLength])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hiddenSize, state_is_tuple=True)
initState = cell.zero_state(batchSize, tf.float32)
outputRNN, state = tf.nn.dynamic_rnn(cell, inputX, initial_state=initState, dtype=tf.float32)
outputRNN = tf.reshape(outputRNN, [-1, hiddenSize])

inputFC = tf.contrib.layers.fully_connected(inputs=outputRNN, num_outputs=numberOfClasses,
activation_fn=None)
inputFC = tf.reshape(inputFC, [batchSize, sequenceLength, numberOfClasses])

weight = tf.ones([batchSize, sequenceLength])
cost = tf.contrib.seq2seq.sequence_loss(logits=inputFC, targets=inputY, weights=weight)
cost = tf.reduce_mean(cost)
train = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

prediction = tf.argmax(inputFC, axis=2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(51):
    costValue, trainValue = sess.run([cost, train],
    feed_dict={inputX: inputOneHot, inputY: labelData})
    predValue = sess.run(prediction, feed_dict={inputX: inputOneHot})
    
    if i % 10 == 0:
        print("{} \t cost : {} \t pred : {}".format(i, costValue, predValue))
        result = [id2char[c] for c in np.squeeze(predValue)]
        print("\t" + str(result))

print("DONE")