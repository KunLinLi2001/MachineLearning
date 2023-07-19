from sklearn import datasets
'''
from sklearn import datasets 是一个Python模块，提供了一些常用的数据集，被广泛应用于机器学习和数据挖掘等领域的训练和测试。
这些数据集都是经过清洗和处理的，因此可以直接用于机器学习算法的训练和评估，而无需进行繁琐的数据清洗和处理。
一些常用的数据集包括：
鸢尾花（iris）数据集：一个经典的用于分类问题的数据集，包括150个样本，每个样本包括四个特征。
手写数字（digits）数据集：一个用于图像分类问题的数据集，包括1797个手写数字图像，每个图像由8x8像素组成。
波士顿房价（boston）数据集：一个用于回归问题的数据集，包括506个样本，每个样本包括13个特征。
乳腺癌（breast_cancer）数据集：一个用于二元分类问题的数据集，包括569个样本，每个样本包括30个特征。
通过导入datasets模块，可以方便地获取这些数据集，例如：

'''


# 获取鸢尾花数据集
iris = datasets.load_iris()
# 获取手写数字数据集
digits = datasets.load_digits()
# 获取波士顿房价数据集
boston = datasets.load_boston()
# 获取乳腺癌数据集
breast_cancer = datasets.load_breast_cancer()

'''
data：特征值 (数组)
target：标签值 (数组)
target_names：标签 (列表)
DESCR：数据集描述
feature_names：特征 (列表)
filename：iris.csv 文件路径
'''
print(iris.keys())
print(type(iris.data)) # numpy矩阵
print(type(iris.target))
print(iris.target_names)
print(type(iris.DESCR))
print(iris.feature_names)
print(iris.filename)