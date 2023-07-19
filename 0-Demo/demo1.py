# 检验Keras是否能正常使用

import tensorflow as tf

# 定义一个简单的线性回归模型
class LinearRegression(tf.Module):
    def __init__(self):
        self.w = tf.Variable(0.0, name='weight')
        self.b = tf.Variable(0.0, name='bias')

    def __call__(self, x):
        return self.w * x + self.b

# 准备训练数据
x_train = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
y_train = tf.constant([2, 4, 6, 8, 10], dtype=tf.float32)

# 定义损失函数和优化器
lr = LinearRegression()
loss_fn = tf.losses.MeanSquaredError()
optimizer = tf.optimizers.SGD(0.01)

# 使用训练数据进行模型训练，并打印训练过程中的损失值
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = lr(x_train)
        loss = loss_fn(y_train, y_pred)

    grads = tape.gradient(loss, lr.variables)
    optimizer.apply_gradients(zip(grads, lr.variables))

    if i % 10 == 0:
        print("Step {}: Loss = {}".format(i, loss))
