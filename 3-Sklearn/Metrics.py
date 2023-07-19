from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 构建模型
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train, y_train)
test_predict = svm.predict(X_test)

from sklearn.metrics import confusion_matrix

y_pred = svm.predict(X_test)
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(conf_mat)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 构建模型
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train, y_train)

'''
Scikit-Learn中的sklearn.metrics模块包含了很多模型评价和度量指标，可以用于对模型的预测进行评估和测试。这些评估方法适用于分类、回归、聚类等一些常见机器学习算法的评估，还有一些专为Python自然语言处理（NLP）和图像处理（Image Processing）而设计的指标。

metrics中一些常用的函数包括：

accuracy_score()：用于计算分类模型的分类准确率；
confusion_matrix()：用于计算分类问题的混淆矩阵；
mean_absolute_error()：用于计算回归模型的平均绝对误差；
mean_squared_error()：用于计算回归模型的均方误差；
r2_score()：用于计算回归模型的决定系数。
这些函数都是很有用的工具，可以帮助我们对机器学习算法生成的模型做出更加准确的评价和判断。
'''
