import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier  # k近邻法
from sklearn.svm import SVC  # SVM分类器类
from sklearn.metrics import accuracy_score

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

'''1.产生两类均值向量、协方差矩阵如下的样本数据'''
mean1, mean2 = [-2, -2], [2, 2]
cov1, cov2 = [[1, 0], [0, 1]], [[1, 0], [0, 4]]

'''2.每类产生500个样本作为训练样本;每类产生100个样本作为测试样本;并随机进行标注'''
# 生成训练样本和测试样本（多元随机正态分布）
train1 = np.random.multivariate_normal(mean1, cov1, 500)
train2 = np.random.multivariate_normal(mean2, cov2, 500)
test1 = np.random.multivariate_normal(mean1, cov1, 100)
test2 = np.random.multivariate_normal(mean2, cov2, 100)
# 合并train和test
train_x = np.concatenate((train1, train2))
test_x = np.concatenate((test1, test2))
# 标注样本类别，类别1表示第一类样本，类别-1表示第二类样本
train_y = np.array([1] * 500 + [-1] * 500)
test_y = np.array([1] * 100 + [-1] * 100)

'''3.画出训练样本和测试样本的分布图'''
plt.figure(1)
plt.scatter(train1[:, 0], train1[:, 1], c='r', label='第一类',s=5)
plt.scatter(train2[:, 0], train2[:, 1], c='b', label='第二类',s=5)
plt.legend()
plt.title('训练样本')
plt.figure(2)
plt.scatter(test1[:, 0], test1[:, 1], c='r', label='第一类',s=5)
plt.scatter(test2[:, 0], test2[:, 1], c='b', label='第二类',s=5)
plt.legend()
plt.title('测试样本')
plt.show()

'''4.按最近邻法用训练样本对测试样本分类,计算平均错误率'''
knn = KNeighborsClassifier(n_neighbors=1) # k=1
np.warnings.filterwarnings('ignore')
knn.fit(train_x, train_y)
test_predict = knn.predict(test_x)
accuracy_knn = accuracy_score(test_y, test_predict)
print('采用最近邻法的准确率: {:.2f}%'.format(accuracy_knn * 100))

'''5.按SVM方法用训练样本对测试样本分类,计算平均错误率;'''
# C：正则化参数，控制对误分类样本的惩罚程度，C越大惩罚越强。
# kernel：核函数类型
# gamma：核系数，控制样本映射到高维空间后的分布
# tol：用于停止训练的误差容忍值
svm = SVC(kernel='rbf',C=1,gamma=0.5,tol=1e-3)
svm.fit(train_x, train_y)
test_predict = svm.predict(test_x)
accuracy_svm = accuracy_score(test_y, test_predict)
print('采用非线性SVM方法的准确率: {:.2f}%'.format(accuracy_svm * 100))

'''6.对两种方法进行对比'''
if accuracy_svm > accuracy_knn:
    print('该数据集中采用SVM准确率更高一些')
elif accuracy_svm < accuracy_knn:
    print('该数据集中采用近邻法准确率更高一些')
else:
    print('该数据集中近邻法和SVM准确率相当')
