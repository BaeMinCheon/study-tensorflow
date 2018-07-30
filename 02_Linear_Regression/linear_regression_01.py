import tensorflow as tf
tf.set_random_seed(777)

sess = tf.Session()

trainX = [1, 2, 3]
trainY = [1, 2, 3]

weight = tf.Variable(tf.random_normal([1]), name='weight')
bias = tf.Variable(tf.random_normal([1]), name='bias')

sess.run(tf.global_variables_initializer())
print('initial value of weight = ' + str(sess.run(weight)))
print('initial value of bias = ' + str(sess.run(bias)))
print()

hypothesis = weight * trainX + bias
cost = tf.reduce_mean(tf.square(hypothesis - trainY))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)

sess.run(tf.global_variables_initializer())
for step in range(2001):
    sess.run(train)
    if step % 100 == 0:
        print(step, sess.run(cost), sess.run(weight), sess.run(bias))