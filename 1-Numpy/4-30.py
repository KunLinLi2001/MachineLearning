import scipy.spatial.distance as dist
import numpy as np
# A和B是两个坐标数组，分别包含10个点
A = np.random.rand(10,2)   # 10个点，每个点有2个坐标值
B = np.random.rand(10,2)   # 10个点，每个点有2个坐标值

d = dist.cdist(A,B)   # 计算A中每个点到B中每个点之间的距离

print(d)   # 输出距离矩阵
