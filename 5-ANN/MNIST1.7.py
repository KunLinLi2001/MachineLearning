import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data", one_hot=True)

batch_size = 100  # 设置每一轮训练的batch的大小
learning_rate = 0.8  # 初始学习率
learning_rate_decay = 0.999  # 学习率的衰减
max_steps = 300000  # 最大训练步数
training_step = tf.Variable(0, trainable=False)


# 定义训练轮数的变量，一般将训练轮数变量的参数设为不可训练的 trainable = False
# 定义得到隐藏层到输出层的向前传播计算方式，激活函数使用relu()  向前传播过程定义为hidden_layer()函数
def hidden_layer(input_tensor, weights1, biases1, weights2, biases2, layer_name):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.matmul(layer1, weights2) + biases2


# x在运行会话是会feed图片数据 y_在会话时会feed答案(label)数据
x = tf.placeholder(tf.float32, [None, 784], name="x-input")
y_ = tf.placeholder(tf.float32, [None, 10], name="y-output")

# 生成隐藏层参数，其中weights包含784*500=392000个参数
weights1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1, shape=[500]))

# 生成输出层参数，其中weights包含50000个参数
weights2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1, shape=[10]))

# y得到了前向传播的结果
y = hidden_layer(x, weights1, biases1, weights2, biases2, 'y')

# 实现一个变量的滑动平均首先需要通过train.ExponentiadlMoving-Average()函数初始化一个滑动平均类，同时需要向函数提供一个衰减率
averages_class = tf.train.ExponentialMovingAverage(0.99, training_step)  # 初始化一个滑动平均类，衰弱率为0.99
# 同时这里也提供了num_updates参数，将其设置为training_step
averages_op = averages_class.apply(tf.trainable_variables())  # 可以通过类函数apply()提供要进行滑动平均计算的变量

# 再次计算经过神经网络前向传播后得到的y值，这里使用了滑动平均，但要牢记滑动平均只是一个影子变量
averages_y = hidden_layer(x, averages_class.average(weights1),
                          averages_class.average(biases1),
                          averages_class.average(weights2),
                          averages_class.average(biases2), 'average_y')

# 交叉熵计算
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
regularization = regularizer(weights1) + regularizer(weights2)
loss = tf.reduce_mean(cross_entropy) + regularization
learning_rate = tf.train.exponential_decay(learning_rate, training_step, mnist.train.num_examples / batch_size,
                                           learning_rate_decay)
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=training_step)

with tf.control_dependencies([training_step, averages_op]):
    train_op = tf.no_op(name="train")
    crorent_predicition = tf.equal(tf.argmax(averages_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(crorent_predicition, tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}
    for i in range(max_steps):
        if i % 1000 == 0:
            validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
            print("After %d training steps,validation accuracy using average model is %g%%" % (
            i, validate_accuracy * 100))
            xs, ys = mnist.train.next_batch(batch_size=100)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
    test_accuracy = sess.run(accuracy, feed_dict=test_feed)
    print("After %d training steps,test accuracy using average model is %g%%" % (max_steps, test_accuracy * 100))