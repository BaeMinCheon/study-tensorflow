import tensorflow as tf
tf.set_random_seed(777)

class Model:
    
    def __init__(self, sess, name, learningRate):
        self.sess = sess
        self.name = name
        self.learningRate = learningRate
        self._build_net()
    
    def _build_net(self):
        self.isTrain = tf.placeholder(tf.bool)
        self.inputX = tf.placeholder(tf.float32, [None, 784])
        self.inputImg = tf.reshape(self.inputX, [-1, 28, 28, 1])
        self.inputY = tf.placeholder(tf.float32, [None, 10])

        conv01 = tf.layers.conv2d(inputs=self.inputImg, filters=32, kernel_size=[3, 3],
        padding='SAME', activation=tf.nn.relu)
        pool01 = tf.layers.max_pooling2d(inputs=conv01, pool_size=[2, 2],
        padding='SAME', strides=2)
        drop01 = tf.layers.dropout(inputs=pool01, rate=0.3, training=self.isTrain)

        conv02 = tf.layers.conv2d(inputs=drop01, filters=64, kernel_size=[3, 3],
        padding='SAME', activation=tf.nn.relu)
        pool02 = tf.layers.max_pooling2d(inputs=conv02, pool_size=[2, 2],
        padding='SAME', strides=2)
        drop02 = tf.layers.dropout(inputs=pool02, rate=0.3, training=self.isTrain)

        conv03 = tf.layers.conv2d(inputs=drop02, filters=128, kernel_size=[3, 3],
        padding='SAME', activation=tf.nn.relu)
        pool03 = tf.layers.max_pooling2d(inputs=conv03, pool_size=[2, 2],
        padding='SAME', strides=2)
        drop03 = tf.layers.dropout(inputs=pool03, rate=0.3, training=self.isTrain)

        flat = tf.reshape(drop03, [-1, 128 * 4 * 4])
        dense = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
        drop04 = tf.layers.dropout(inputs=dense, rate=0.5, training=self.isTrain)

        self.logit = tf.layers.dense(inputs=drop04, units=10)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logit,
        labels=self.inputY))
        self.train = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.cost)

        self.prediction = tf.equal(tf.argmax(self.logit, 1), tf.argmax(self.inputY, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.prediction, tf.float32))

    def getPredict(self, testX, isTrain=False):
        return self.sess.run(self.logit,
        feed_dict={self.inputX: testX, self.isTrain: isTrain})

    def getAccuracy(self, testX, testY, isTrain=False):
        return self.sess.run(self.accuracy,
        feed_dict={self.inputX: testX, self.inputY: testY, self.isTrain: isTrain})

    def processTrain(self, dataX, dataY, isTrain=True):
        return self.sess.run([self.cost, self.train],
        feed_dict={self.inputX: dataX, self.inputY: dataY, self.isTrain: isTrain})
