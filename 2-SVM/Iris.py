from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import sklearn.metrics as met
from matplotlib import pyplot as plt

'''1.加载python自带的鸢尾花数据集'''
iris=datasets.load_iris()
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
# print(iris.keys())
X = iris.data # 数据集
y = iris.target # 类别

'''2.划分训练集和测试集'''
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

'''3.构建SVM模型并手动调节超参数'''
'''4.模型预测，并利用混淆矩阵查看预测错误的位置'''
'''
混淆矩阵中行表示真实类别，列表示预测类别，
对角线上的数字表示预测正确的样本数，
非对角线上的数字表示预测错误的样本数。
混淆矩阵中第二行第三列的1表示有一个样本实际上是第三类，结果被预测成第二类
'''
# （1）构建线性模型
svm = SVC(kernel='linear',C=1,gamma=0.5,tol=1e-3) # 构建模型
svm.fit(X_train, y_train) # 训练
y_pred = svm.predict(X_test) # 测试
accuracy_svm = met.accuracy_score(y_test, y_pred) # 准确率
conf_mat = met.confusion_matrix(y_true=y_test, y_pred=y_pred) # 混淆矩阵
print("采用线性svm的准确率为{:.2f}%".format(accuracy_svm*100))
print("混淆矩阵如下：")
print(conf_mat)

# （2）构建二次项模型
svm = SVC(kernel='poly',C=1,degree=2,gamma=0.5,tol=1e-3)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy_svm = met.accuracy_score(y_test, y_pred)
conf_mat = met.confusion_matrix(y_true=y_test, y_pred=y_pred)
print("采用二次项svm的准确率为{:.2f}%".format(accuracy_svm*100))
print("混淆矩阵如下：")
print(conf_mat)

# （3）构建三次项模型
svm = SVC(kernel='poly',C=1,degree=3,gamma=0.5,tol=1e-3)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy_svm = met.accuracy_score(y_test, y_pred)
conf_mat = met.confusion_matrix(y_true=y_test, y_pred=y_pred)
print("采用三次项svm的准确率为{:.2f}%".format(accuracy_svm*100))
print("混淆矩阵如下：")
print(conf_mat)

# （4）构建六次项模型
svm = SVC(kernel='poly',C=1,degree=6,gamma=0.5,tol=1e-3)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy_svm = met.accuracy_score(y_test, y_pred)
conf_mat = met.confusion_matrix(y_true=y_test, y_pred=y_pred)
print("采用六次项svm的准确率为{:.2f}%".format(accuracy_svm*100))
print("混淆矩阵如下：")
print(conf_mat)

# （5）构建基于smo算法的模型
svm = SVC(kernel='rbf',C=1,gamma=0.5,tol=1e-3)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy_svm = met.accuracy_score(y_test, y_pred)
conf_mat = met.confusion_matrix(y_true=y_test, y_pred=y_pred)
print("采用基于SMO算法的svm的准确率为{:.2f}%".format(accuracy_svm*100))
print("混淆矩阵如下：")
print(conf_mat)

# （5）构建sigmoid模型
svm = SVC(kernel='sigmoid',C=1,gamma=0.5,tol=1e-3)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy_svm = met.accuracy_score(y_test, y_pred)
conf_mat = met.confusion_matrix(y_true=y_test, y_pred=y_pred)
print("采用基于sigmoid的svm的准确率为{:.2f}%".format(accuracy_svm*100))
print("混淆矩阵如下：")
print(conf_mat)



