import tensorflow as tf
tf.set_random_seed(777)
import numpy as np

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

charSet = list(set(sentence))
charDic = {w : i for i, w in enumerate(charSet)}

inputDim = len(charSet)
hiddenSize = len(charSet)
numberOfClasses = len(charSet)
sequenceLength = 10             # window size
learningRate = 0.1

dataX = []
dataY = []
for i in range(0, len(sentence) - sequenceLength):
    strX = sentence[i : i + sequenceLength]
    strY = sentence[i + 1 : i + sequenceLength + 1]
    print(i, strX, ' -> ', strY)

    dataX.append([charDic[c] for c in strX])
    dataY.append([charDic[c] for c in strY])

batchSize = len(dataX)

inputX = tf.placeholder(tf.int32, [None, sequenceLength])
inputY = tf.placeholder(tf.int32, [None, sequenceLength])
inputOneHot = tf.one_hot(inputX, numberOfClasses)
print(inputOneHot)

def GetLSTM():
    cell = tf.contrib.rnn.BasicLSTMCell(hiddenSize, state_is_tuple=True)
    return cell
cell = tf.contrib.rnn.MultiRNNCell([GetLSTM() for i in range(2)], state_is_tuple=True)
outputRNN, state = tf.nn.dynamic_rnn(cell, inputOneHot, dtype=tf.float32)
outputRNN = tf.reshape(outputRNN, [-1, hiddenSize])

inputFC = tf.contrib.layers.fully_connected(outputRNN, numberOfClasses, activation_fn=None)
inputFC = tf.reshape(inputFC, [batchSize, sequenceLength, numberOfClasses])

weight = tf.ones([batchSize, sequenceLength])
cost = tf.contrib.seq2seq.sequence_loss(logits=inputFC, targets=inputY, weights=weight)
cost = tf.reduce_mean(cost)
train = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(501):
    sess.run(train, feed_dict={inputX: dataX, inputY: dataY})

test = sess.run(inputFC, feed_dict={inputX: dataX})
for i, result in enumerate(test):
    index = np.argmax(result, axis=1)

    if i is 0:
        print(''.join([charSet[t] for t in index]), end='')
    else:
        print(charSet[index[-1]], end='')

print()
print("DONE")