import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# learningRate = float(input("input learning rate : "))
learningRate = 0.1

dataIn = [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]] # 4x2
dataOut = [[-1.0], [1.0], [1.0], [-1.0]] # 4x1

AB = tf.placeholder(tf.float32, [4, 2])
F = tf.placeholder(tf.float32, [4, 1])

weight01 = tf.Variable([[1.0, 1.0], [1.0, 1.0]]) # 2x2
bias01 = tf.Variable([1.0, 1.0]) # 2x0
output01 = tf.tanh(tf.add(tf.matmul(AB, weight01), bias01)) # 4x2

weight02 = tf.Variable([[1.0], [1.0]]) # 2x1
bias02 = tf.Variable([1.0]) # 1x0
output02 = tf.tanh(tf.add(tf.matmul(output01, weight02), bias02)) # 4x1

error = tf.divide(tf.reduce_mean(tf.square(tf.subtract(output02, dataOut))), 2.0)

#        m1       a1       f1        m2       a2       f2        err
# AB -> (mul) -> (add) -> (tanh) -> (mul) -> (add) -> (tanh) -> (error)
#        ^        ^                  ^        ^
#        |        |                  |        |
#        w1       b1                 w2       b2

devErr = tf.subtract(output02, dataOut) # 4x1
devF02 = tf.multiply(devErr, tf.subtract(1.0, tf.square(output02)))
devA02 = devF02 # 4x1

devWeight02 = tf.matmul(tf.transpose(output01), devA02) # 2x1 = 2x4 * 4x1
devWeight02 = tf.divide(devWeight02, tf.cast(tf.shape(output01)[0], dtype=tf.float32)) # shape(output01)[0] = 4
devBias02 = tf.reduce_mean(devA02, axis=[0]) # 1x0 

devM02 = tf.matmul(devA02, tf.transpose(weight02)) # 4x2 = 4x1 * 1x2
devF01 = tf.multiply(devM02, tf.subtract(1.0, tf.square(output01)))
devA01 = devF01 # 4x2

devWeight01 = tf.matmul(tf.transpose(AB), devA01) # 2x2 = 2x4 * 4x2
devWeight01 = tf.divide(devWeight01, tf.cast(tf.shape(AB)[0], dtype=tf.float32)) # shape(AB)[0] = 4
devBias01 = tf.reduce_mean(devA01, axis=[0]) # 2x0

train = [
    tf.assign(weight01, tf.subtract(weight01, tf.multiply(devWeight01, learningRate))),
    tf.assign(bias01, tf.subtract(bias01, tf.multiply(devBias01, learningRate))),
    tf.assign(weight02, tf.subtract(weight02, tf.multiply(devWeight02, learningRate))),
    tf.assign(bias02, tf.subtract(bias02, tf.multiply(devBias02, learningRate)))
]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1001):
        errorV, trainV, outV = sess.run([error, train, output02], feed_dict={AB: dataIn, F: dataOut})
        
        # print("train number #{}".format(i))
        # print(" error : {}".format(errorV))
        # print()
        
        if i % 100 == 0:
            print("train number #{}".format(i))
            print(" error : {}".format(errorV))
            print(" output \n{}".format(outV))
            print()

print("\n \t TRAIN DONE")