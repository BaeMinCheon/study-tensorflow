import tensorflow as tf
tf.set_random_seed(777)
import numpy as np
train_data = np.array(
    [[828.659973, 833.450012, 908100, 828.349976, 831.659973],
    [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
    [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
    [816, 820.958984, 1008100, 815.48999, 819.23999],
    [819.359985, 823, 1188100, 818.469971, 818.97998],
    [819, 823, 1198100, 816, 820.450012],
    [811.700012, 815.25, 1098100, 809.780029, 813.669983],
    [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

is_regulating = int(input("do u wanna rescale data ? (1 for yes, 0 for no) : "))
if is_regulating == 1:
    train_data = MinMaxScaler(train_data)

trainX = train_data[:, 0:-1]
trainY = train_data[:, [-1]]

inputX = tf.placeholder(tf.float32, [None, 4])
inputY = tf.placeholder(tf.float32, [None, 1])

weight = tf.Variable(tf.random_normal([4, 1]))
bias = tf.Variable(tf.random_normal([1]))

hypothesis = tf.matmul(inputX, weight) + bias
cost = tf.reduce_mean(tf.square(hypothesis - inputY))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(101):
    trainValue = sess.run(train, feed_dict={inputX: trainX, inputY: trainY})
    if step % 10 == 0:
        costValue, weightValue = sess.run([cost, weight],
        feed_dict={inputX: trainX, inputY: trainY})
        print("step : #{:4} \t cost : {} \t weight : {}".format(step, costValue, weightValue))

print("DONE")