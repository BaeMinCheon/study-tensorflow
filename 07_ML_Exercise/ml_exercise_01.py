import tensorflow as tf
tf.set_random_seed(777)

learning_rate = float(input("input learning rate : "))

trainX = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
trainY = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]
testX = [[2, 1, 1],
         [3, 1, 2],
         [3, 3, 4]]
testY = [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 1]]

inputX = tf.placeholder(tf.float32, [None, 3])
inputY = tf.placeholder(tf.float32, [None, 3])

weight = tf.Variable(tf.random_normal([3, 3]))
bias = tf.Variable(tf.random_normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(inputX, weight) + bias)
cost = tf.reduce_mean(tf.reduce_sum(-inputY * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

prediction = tf.arg_max(hypothesis, 1)
isCorrect = tf.equal(prediction, tf.arg_max(inputY, 1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(201):
    sess.run(train, feed_dict={inputX: trainX, inputY: trainY})
    if step % 20 == 0:
        costValue, weightValue = sess.run([cost, weight],
        feed_dict={inputX: trainX, inputY: trainY})
        print("step : #{:4} \t cost : {} \t weight : {}".format(step, costValue, weightValue))

print("prediction : ", sess.run(prediction, feed_dict={inputX: testX}))
print("accuracy : ", sess.run(accuracy, feed_dict={inputX: testX, inputY: testY}))

print("DONE")