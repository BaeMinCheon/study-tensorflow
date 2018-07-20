import tensorflow as tf
tf.set_random_seed(777)

trainX = [
    [73.0, 80.0, 75.0],
    [93.0, 88.0, 93.0], 
    [89.0, 91.0, 90.0], 
    [96.0, 98.0, 100.0], 
    [73.0, 66.0, 70.0]
]
trainY = [
    [152.0], 
    [185.0], 
    [180.0], 
    [196.0], 
    [142.0]
]

inputX = tf.placeholder(tf.float32, shape=[None, 3])
inputY = tf.placeholder(tf.float32, shape=[None, 1])

weight = tf.Variable(tf.random_normal([3, 1]), name='weight')
bias = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(inputX, weight) + bias
print("hypo -> ", hypothesis)
cost = tf.reduce_mean(tf.square(hypothesis - inputY))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess01 = tf.Session()
sess01.run(tf.global_variables_initializer())
for step in range(2001):
    costValue, hypoValue, trainValue = sess01.run([cost, hypothesis, train],
    feed_dict={inputX: trainX, inputY: trainY})
    if step % 50 == 0:
        print(step, "( cost, hypo ) -> ", costValue, hypoValue)

print("DONE")