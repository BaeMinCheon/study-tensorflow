import tensorflow as tf
tf.set_random_seed(777)

trainX = tf.placeholder(tf.float32, shape=[None])
trainY = tf.placeholder(tf.float32, shape=[None])

weight = tf.Variable(tf.random_normal([1]), name='weight')
bias = tf.Variable(tf.random_normal([1]), name='bias')

sess01 = tf.Session()
sess01.run(tf.global_variables_initializer())
print('initial value of weight = ' + str(sess01.run(weight)))
print('initial value of bias = ' + str(sess01.run(bias)))
print()

hypothesis = weight * trainX + bias
cost = tf.reduce_mean(tf.square(hypothesis - trainY))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess02 = tf.Session()
sess02.run(tf.global_variables_initializer())
for step in range(2001):
    costValue, weightValue, biasValue, trainValue = sess02.run([cost, weight, bias, train], feed_dict={trainX: [1, 2, 3], trainY: [1, 2, 3]})
    if step % 20 == 0:
        print(step, costValue, weightValue, biasValue)