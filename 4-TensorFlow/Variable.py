import tensorflow as tf

'''一、创建变量'''
# 要创建变量，请提供一个初始值。tf.Variable 与初始值的 dtype 相同。
my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
my_variable = tf.Variable(my_tensor)
# Variables can be all kinds of types, just like tensors
bool_variable = tf.Variable([False, False, False, True])
complex_variable = tf.Variable([5 + 4j, 6 + 1j])
# 变量与张量的定义方式和操作行为都十分相似，实际上，它们都是 tf.Tensor 支持的一种数据结构。
# 与张量类似，变量也有 dtype 和形状，并且可以导出至 NumPy。
print("Shape: ", my_variable.shape)
print("DType: ", my_variable.dtype)
print("As NumPy: ", my_variable.numpy())

'''二、变量运算'''
# 大部分张量运算在变量上也可以按预期运行，不过变量无法重构形状。
print("A variable:", my_variable)
print("\nViewed as a tensor:", tf.convert_to_tensor(my_variable))
print("\nIndex of highest value:", tf.math.argmax(my_variable))
# This creates a new tensor; it does not reshape the variable.
print("\nCopying and reshaping: ", tf.reshape(my_variable, [1,4]))

'''三、重分配变量'''
# 如上所述，变量由张量提供支持。您可以使用 tf.Variable.assign 重新分配张量。
# 调用 assign（通常）不会分配新张量，而会重用现有张量的内存。
a = tf.Variable([2.0, 3.0])
# This will keep the same dtype, float32
a.assign([1, 2])
# Not allowed as it resizes the variable:
try:
  a.assign([1.0, 2.0, 3.0])
except Exception as e:
  print(f"{type(e).__name__}: {e}")

# 如果在运算中像使用张量一样使用变量，那么通常会对支持张量执行运算。
# 从现有变量创建新变量会复制支持张量。两个变量不能共享同一内存空间。
a = tf.Variable([2.0, 3.0])
# Create b based on the value of a
b = tf.Variable(a)
a.assign([5, 6])

# a and b are different
print(a.numpy())
print(b.numpy())

# There are other versions of assign
print(a.assign_add([2,3]).numpy())  # [7. 9.]
print(a.assign_sub([7,9]).numpy())  # [0. 0.]

'''四、生命周期、命名和监视'''
# 在基于 Python 的 TensorFlow 中，tf.Variable 实例与其他 Python 对象的生命周期相同。
# 如果没有对变量的引用，则会自动将其解除分配。
a = tf.Variable(my_tensor, name="Mark")
# A new variable with the same name, but different value
# Note that the scalar add is broadcast
b = tf.Variable(my_tensor + 1, name="Mark")
# These are elementwise-unequal, despite having the same name
print(a == b)
# 虽然变量对微分很重要，但某些变量不需要进行微分。
# 在创建时，通过将 trainable 设置为 False 可以关闭梯度。
# 例如，训练计步器就是一个不需要梯度的变量。
step_counter = tf.Variable(1, trainable=False)
print(step_counter)

'''五、放置变量和张量'''
# 如果在有 GPU 和没有 GPU 的不同后端上运行此笔记本，则会看到不同的记录。
with tf.device('CPU:0'):

  # Create some tensors
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  c = tf.matmul(a, b)

print(c)
# 如果您有多个 GPU 工作进程，但希望变量只有一个副本，则可以这样做
with tf.device('CPU:0'):
  a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.Variable([[1.0, 2.0, 3.0]])
with tf.device('GPU:0'):
  # Element-wise multiply
  k = a * b
print(k)



























