import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 不输出Info

'''一、创建一些基本张量'''
print('一、创建一些基本张量')
# 1.“标量”（或称“0 秩”张量）。标量包含单个值，但没有“轴”。
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)
# 2.“向量”（或称“1 秩”张量）就像一个值列表。向量有 1 个轴：
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)
# 3.“矩阵”（或称“2 秩”张量）有 2 个轴：
rank_2_tensor = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float16)
print(rank_2_tensor)
# 4.张量的轴可能更多，下面是一个包含 3 个轴的张量：
rank_3_tensor = tf.constant([
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
    [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
    [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
])
print(rank_3_tensor)

'''二、张量的转换'''
print('二、张量的转换')
# 通过使用 np.array 或 tensor.numpy 方法，将张量转换为 NumPy 数组：
arr1 = np.array(rank_2_tensor)
print(arr1)
print(type(arr1))
arr2 = rank_2_tensor.numpy()
print(arr2)
print(type(arr2))

'''三、张量的运算'''
print('三、张量的运算')
# 1.可以对张量执行基本数学运算
print('1.可以对张量执行基本数学运算:')
a = tf.constant([[1, 2],[3, 4]])
b = tf.ones([2,2],dtype=tf.int32) # 必须类型要相同
print(tf.add(a, b), "\n") # 加法
print(tf.multiply(a, b), "\n") # 点乘
print(tf.matmul(a, b), "\n") # 矩阵乘法
print(a + b, "\n") # 加法
print(a * b, "\n") # 点乘
print(a @ b, "\n") # 矩阵乘法
# 2.各种运算都可以使用张量。
print('2.各种运算都可以使用张量:')
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])
print(tf.reduce_max(c)) # 最大值
print(tf.math.argmax(c)) # 最大值下角标
print(tf.nn.softmax(c)) # softmax
# 3.转换成张量
print('3.转换成张量:')
d = np.array([1,2,3])
print(tf.convert_to_tensor(d))
print(tf.reduce_max(d))
print(tf.reduce_max([1,2,3]))

'''四、张量形状简介'''
# 4 秩张量，形状：[3, 2, 4, 5]
rank_4_tensor = tf.zeros([3, 2, 4, 5])
print(rank_4_tensor)
print("Type of every element:", rank_4_tensor.dtype) # 数据类型
print("Number of axes:", rank_4_tensor.ndim) # 秩4
print("Shape of tensor:", rank_4_tensor.shape) # 形状(3, 2, 4, 5)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0]) # 第一个轴3
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1]) # 最后的轴
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy()) # 大小
# Tensor.ndim 和 Tensor.shape 特性不返回 Tensor 对象。
# 如果需要 Tensor，请使用 tf.rank 或 tf.shape 函数。
print(tf.rank(rank_4_tensor)) # 秩
print(tf.shape(rank_4_tensor)) # 形状
# 虽然通常用索引来指代轴，但是您始终要记住每个轴的含义。
# 轴一般按照从全局到局部的顺序进行排序：首先是批次轴，随后是空间维度（长宽），最后是每个位置的特征。
# 这样，在内存中，特征向量就会位于连续的区域。

'''五、张量索引'''
# 1.单轴索引
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())
# 使用标量编制索引会移除轴：
print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())
# 使用 : 切片编制索引会保留轴：
print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())

# 2.多轴索引
# 对于高秩张量的每个单独的轴，遵循与单轴情形完全相同的规则。
print(rank_2_tensor.numpy())
# 为每个索引传递一个整数，结果是一个标量。
print(rank_2_tensor[1, 1].numpy())
# 使用整数与切片的任意组合编制索引：
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")
# 3轴张量的示例
print(rank_3_tensor[:, :, 4])

'''# 六、操作形状'''
x = tf.constant([[1], [2], [3]])
print(x.shape)
print(x)
# 可以直接转换为list
print(x.shape.as_list())
# 通过重构可以改变张量的形状。tf.reshape 运算的速度很快，资源消耗很低，因为不需要复制底层数据。
reshaped = tf.reshape(x, [1, 3])
print(x.shape)
print(reshaped.shape)
# 展平张量，则可以看到它在内存中的排列顺序。
print(tf.reshape(rank_3_tensor, [-1]))
# 一般来说，tf.reshape 唯一合理的用途是用于合并或拆分相邻轴（或添加/移除 1）。
# 对于 3x2x5 张量，重构为 (3x2)x5 或 3x(2x5) 都合理，因为切片不会混淆：
print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, -1]))
print('------------------------------------------')

# 重构可以处理总元素个数相同的任何新形状，但是如果不遵从轴的顺序，则不会发挥任何作用。
# 利用 tf.reshape 无法实现轴的交换;交换轴，您需要使用 tf.transpose
print(tf.reshape(rank_3_tensor, [2, 3, 5]), "\n")
# This is a mess
print(tf.reshape(rank_3_tensor, [5, 6]), "\n")
# This doesn't work at all
try:
  tf.reshape(rank_3_tensor, [7, -1])
except Exception as e:
  print(f"{type(e).__name__}: {e}")

'''七、DTypes 详解'''
print('七、DTypes 详解')
# 使用 Tensor.dtype 属性可以检查 tf.Tensor 的数据类型。
# 从 Python 对象创建 tf.Tensor 时，您可以选择指定数据类型。
# 如果不指定，TensorFlow 会选择一个可以表示您的数据的数据类型。
# TensorFlow 将 Python 整数转换为 tf.int32，
# 将 Python 浮点数转换为 tf.float32。
# 另外，当转换为数组时，TensorFlow 会采用与 NumPy 相同的规则。
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
# 转换类型
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
print(the_u8_tensor)

'''八、广播'''
x = tf.constant([1, 2, 3])
y = tf.constant(2)
z = tf.constant([2, 2, 2])
# 实现结果相同
print(tf.multiply(x, 2))
print(x * y)
print(x * z)
# 实现结果相同
x = tf.reshape(x,[3,1])
y = tf.range(1, 5)
print(x, "\n")
print(y, "\n")
print(tf.multiply(x, y))
# 在大多数情况下，广播的时间和空间效率更高，因为广播运算不会在内存中具体化扩展的张量。
# 使用 tf.broadcast_to 可以了解广播的运算方式。
print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))

'''九、不规则张量'''
# 如果张量的某个轴上的元素个数可变，则称为“不规则”张量。
# 对于不规则数据，请使用 tf.ragged.RaggedTensor。
ragged_list = [[0, 1, 2, 3],[4, 5],[6, 7, 8],[9]]
try:
  tensor = tf.constant(ragged_list)
except Exception as e:
  print(f"{type(e).__name__}: {e}")
# 应使用 tf.ragged.constant 来创建 tf.RaggedTensor：
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)
# tf.RaggedTensor 的形状将包含一些具有未知长度的轴：
print(ragged_tensor.shape)

'''十、字符串张量'''
print('十、字符串张量')
# Tensors can be strings, too here is a scalar string.
scalar_string_tensor = tf.constant("Gray wolf")
print(scalar_string_tensor)
# If you have three string tensors of different lengths, this is OK.
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
# Note that the shape is (3,). The string length is not included.
# 在 Python3 中，字符串被视为 Unicode 字符串，即每个字符都是 Unicode 编码。
# 而在 TensorFlow 中，字符串类型的张量是使用字节字符串表示（Byte String），
# 即每个字符是 8 比特的字节表示。因此，当输出字符串类型的张量时，
# 需要以字节字符串的格式输出，这时就会在字符串前面加上 "b" 标识字节字符串。
print(tensor_of_strings)
# 如果传递 Unicode 字符，则会使用 utf-8 编码。
print(tf.constant("🥳👍"))
# 在 tf.strings 中可以找到用于操作字符串的一些基本函数，包括 tf.strings.split。
print(tf.strings.split(scalar_string_tensor, sep=" "))
# ...but it turns into a `RaggedTensor` if you split up a tensor of strings,
# as each string might be split into a different number of parts.
print(tf.strings.split(tensor_of_strings))
# 以及 tf.string.to_number：
text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))
# 虽然不能使用 tf.cast 将字符串张量转换为数值，但是可以先将其转换为字节，然后转换为数值。
# tf.io 模块包含在数据与字节类型之间进行相互转换的函数，包括解码图像和解析 csv 的函数。
byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)
# 关于Unicode类型字符的分割与解码
unicode_bytes = tf.constant("アヒル 🦆")
unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, "UTF-8")
unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")
print("\nUnicode bytes:", unicode_bytes)
print("\nUnicode chars:", unicode_char_bytes)
print("\nUnicode values:", unicode_values)

'''十一、稀疏张量'''
# 在某些情况下，数据很稀疏，比如说在一个非常宽的嵌入空间中。
# 为了高效存储稀疏数据，TensorFlow 支持 tf.sparse.SparseTensor 和相关运算。
# Sparse tensors store values by index in a memory-efficient manner
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor, "\n")
# You can convert sparse tensors to dense
print(tf.sparse.to_dense(sparse_tensor))






