import os
import tensorflow as tf
import cProfile

'''1.在 Tensorflow 2.0 中，默认启用 Eager Execution。'''
tmp = tf.executing_eagerly()
print('tf.executing_eagerly() is {}'.format(tmp))

'''2.现在可以运行 TensorFlow 运算，结果将立即返回：'''
x = [[2.]]
# 在 TensorFlow 中，矩阵乘法是通过 tf.matmul() 函数实现的。
# tf.matmul() 可以对两个张量（也就是多维数组）进行乘法运算，其返回值也是一个张量。
m = tf.matmul(x, x)
print("hello, {}".format(m))

'''3.启用 Eager Execution 会改变 TensorFlow 运算的行为方式,现在它们会立即评估并将值返回给 Python。
tf.Tensor 对象会引用具体值，而非指向计算图中节点的符号句柄。
由于无需构建计算图并稍后在会话中运行，可以轻松使用 print() 或调试程序检查结果。
评估、输出和检查张量值不会中断计算梯度的流程。
Eager Execution 可以很好地配合 NumPy 使用。NumPy 运算接受 tf.Tensor 参数。
TensorFlow tf.math 运算会将 Python 对象和 NumPy 数组转换为 tf.Tensor 对象
。tf.Tensor.numpy 方法会以 NumPy ndarray 的形式返回该对象的值。
'''
# 一个张量
a = tf.constant([[1, 2],
                 [3, 4]])
print(a)
