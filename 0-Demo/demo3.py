# 检验TensorFlow是否正常调用GPU
import tensorflow as tf
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 不输出Info

# 创建一个随机输入张量
input_shape = [5000, 5000]  # 定义输入张量的形状
random_input = tf.random.normal(input_shape)

# 使用CPU进行计算
with tf.device('/CPU:0'):
    start_time = time.time()
    for i in range(10):
        # 对输入张量进行数学运算
        result = tf.math.square(random_input)
    print("Time cost on CPU: ", time.time() - start_time)

# 使用GPU进行计算
with tf.device('/GPU:0'):
    start_time = time.time()
    for i in range(10):
        # 对输入张量进行数学运算
        result = tf.math.square(random_input)
    print("Time cost on GPU: ", time.time() - start_time)
