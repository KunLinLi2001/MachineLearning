import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

'''1.随机生成两类样本数据'''
mean1 = np.array([1, 2])  # 样本数据的平均值
cov1 = np.array([[1, 0.5], [0.5, 1]])  # 协方差矩阵
mean2 = np.array([-1, -2])
cov2 = np.array([[1, 0.5], [0.5, 1]])

Sum = 500
x1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=Sum)  # 正态分布生成500个随机样本数据
x2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=Sum)

'''2.为两类样本标记分类'''
y1 = np.ones((Sum,))  # 第一类样本的类别标签y都为1
y2 = -np.ones((Sum,))  # 第二类样本的类别标签y都为-1
x = np.concatenate((x1, x2), axis=0)  # 将两类样本数据并在一起
y = np.concatenate((y1, y2), axis=0)  # 将两类样本数据的类别标签并在一起

'''3.可视化样本数据'''
plt.figure(1)  # 创建一个新的图形窗口
plt.scatter(x1[:, 0], x1[:, 1], c='b', s=5)  # 绘制第一类样本
plt.scatter(x2[:, 0], x2[:, 1], c='r', s=5)  # 绘制第二类样本
plt.title('待训练数据集')

'''4.定义线性SVM'''
# 初始化参数
w = np.zeros(x.shape[1])  # 参数w初始化为全零向量，长度为样本特征的维数
b = 0  # 参数b初始化为0
num_iteration = 10000  # 迭代次数
learning_rate = 0.01  # 学习率

# 定义线性SVM模型
def svm(x, w, b):
    return np.dot(x, w) + b

# 定义损失函数
def loss(x, y, w, b):
    N = x.shape[0]  # 样本数量
    margin = y * svm(x, w, b)  # 计算每个样本到分离超平面的距离，并与样本类别标签相乘
    loss = 1 - margin  # 计算损失
    loss[loss < 0] = 0  # 如果距离大于等于1，则损失为0；否则损失为1-距离
    return np.sum(loss) / N  # 计算平均损失

# 计算梯度
def grad(x, y, w, b):
    N = x.shape[0]  # 样本数量
    margin = y * svm(x, w, b)  # 计算每个样本到分离超平面的距离，并与样本类别标签相乘
    grad_w = np.zeros(x.shape[1])  # 初始化w的梯度为全零向量，长度等于样本特征的维数
    grad_b = 0  # 初始化b的梯度为0
    for i in range(N):
        if margin[i] < 1:  # 如果距离小于1，即样本被分类错误
            grad_w += -y[i] * x[i]  # 更新w的梯度
            grad_b += -y[i]  # 更新b的梯度
    grad_w /= N  # 计算w的平均梯度
    grad_b /= N  # 计算b的平均梯度
    return grad_w, grad_b  # 返回梯度

'''5.迭代训练'''
for i in range(num_iteration):
    # 计算梯度
    grad_w, grad_b = grad(x, y, w, b)
    # 更新参数
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b
    # 计算损失
    train_loss = loss(x, y, w, b)
    if i % 500 == 0:
        print('迭代次数 {}/{}: 目前损失率 = {}'.format(i, num_iteration, train_loss))

'''6.绘制超平面'''
plt.figure(2)  # 创建一个新的图形窗口
plt.scatter(x1[:, 0], x1[:, 1], c='b', s=10)  # 绘制第一类样本
plt.scatter(x2[:, 0], x2[:, 1], c='r', s=10)  # 绘制第二类样本
x_axis = np.linspace(-5, 5, 100)  # 在x轴上生成100个点
y_axis = -(w[0] * x_axis + b) / w[1]  # 计算超平面在x轴上的对应点
plt.plot(x_axis, y_axis, c='g')  # 绘制超平面
plt.title('线性SVM处理结果')  # 设置图像标题

'''7.计算准确率'''
y_pred = np.sign(svm(x, w, b))  # 预测样本类别
accuracy = np.mean(np.equal(y_pred, y))  # 计算准确率
print('最终训练准确率 = {}'.format(accuracy))  # 打印准确率

plt.show()  # 显示图像
