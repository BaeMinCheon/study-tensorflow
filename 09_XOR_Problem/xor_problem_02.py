import tensorflow as tf
tf.set_random_seed(777)

trainX = [
    [1.0],
    [2.0],
    [3.0]
]
trainY = [
    [1.0],
    [2.0],
    [3.0]
]

inputX = tf.placeholder(tf.float32, shape=[None, 1])
inputY = tf.placeholder(tf.float32, shape=[None, 1])

weight = tf.Variable(tf.truncated_normal([1, 1]))
bias = tf.Variable(5.0)

hypothesis = tf.matmul(inputX, weight) + bias
difference = hypothesis - inputY

diffOfLayer01 = difference
diffOfBias = difference
diffOfWeight = tf.matmul(tf.transpose(inputX), diffOfLayer01)

print("inputX : {} \t weight : {} \t devOfLayer01 : {} \t devOfWeight : {}".format(
    inputX, weight, diffOfLayer01, diffOfWeight))

learningRate = 0.1
step = [
    tf.assign(weight, weight - learningRate * diffOfWeight),
    tf.assign(bias, bias - learningRate * tf.reduce_mean(diffOfBias))
]
# without optimizer functions
train = tf.reduce_mean(tf.square(hypothesis - inputY))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(1001):
    stepValue, trainValue = sess.run([step, train],
    feed_dict={inputX: trainX, inputY: trainY})
    if i % 100 == 0:
        print("step #{:4} \t curr : {} \t hypo : {}".format(
            i, stepValue, trainValue
        ))

print(sess.run(hypothesis, feed_dict={inputX: trainX}))