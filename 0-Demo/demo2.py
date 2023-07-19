# 检验TensorFlow是否正确安装

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 不输出Info

tf.compat.v1.disable_eager_execution()
hello = tf.constant('hello,tensorf')
sess = tf.compat.v1.Session()
print(sess.run(hello))
