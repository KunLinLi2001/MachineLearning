from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 载入数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

'''
SVC（Support Vector Classification）是一种支持向量机算法，它可以用于分类任务。在Scikit-Learn中，可以使用sklearn.svm.SVC来创建SVC模型。
SVC是一种二分类模型，它的目标是找到一个能够最大化边界值（margin）的决策边界，将两个不同类别的样本分开。SVC模型中的“支持向量”指的是离决策边界最近的样本点。
SVC通过对样本点找到合适的权重和偏置量，可以通过某种核函数将样本点映射到更高维的特征空间进行分类，最终找到能够使间隔最大化的决策边界。
在使用SVC模型时，需要根据实际情况进行合适的参数调优，包括C（正则化参数）、kernel（核函数）、degree（多项式核函数的度数）、gamma（核系数）等。
Scikit-Learn提供了GridSearchCV等函数帮助我们通过交叉验证进行超参数的搜索和选择。
'''

# 训练SVC模型
svc = SVC(kernel='rbf', C=1, gamma=0.1)
svc.fit(X_train, y_train)

# 测试SVC模型
y_pred = svc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

'''
Scikit-Learn中，可以使用sklearn.svm.SVC来创建SVC模型。SVC模型有多个参数可以用于调整算法的性能，下面是一些常用参数的定义和用法：
C：正则化参数，控制对误分类样本的惩罚程度，C越大惩罚越强。默认值为1.0。较小的C值会导致更多的松弛，即多些数据点将被错误分类；较大的C值会尽量让更多的数据点被分对，但有可能是过度拟合（overfitting）。
kernel：核函数类型，用于将数据映射到高维空间中，通过找到一个合适的决策边界来实现分类。有线性核、多项式核、径向基函数（RBF）核等常用的核函数。默认值为'rbf'（径向基函数）。
degree：多项式核函数的度数，仅在kernel=‘poly’时使用。默认值为3。
gamma：核系数，控制样本映射到高维空间后的分布，如果'gamma'取“auto”，则gamma=1/n，其中n为特征数。默认值为'auto'。
coef0：核函数中的常数项。仅在线性核和’poly’核中有用。默认值为0。
shrinking：是否使用启发式方法计算决策函数来加快计算速度。默认为True。
probability：是否启用类别概率估计。如果训练集数量很大，开启此参数会使训练时间大大增加。默认为False。
tol：用于停止训练的误差容忍值。默认值为1e-3。
cache_size：在内存中缓存核矩阵以加快计算速度。默认为200MB。
class_weight：可以选择类别的加权方式。默认为None，也可以选择对类别平衡的情况进行加权（使用balanced选项）。
verbose：控制在训练过程中的详细程度。默认为0，即不输出任何训练信息。
max_iter：内部迭代次数的最大值。如果不设置，默认值为-1。
'''
'''
SVC中可以传入的核函数为：'linear', 'poly', 'rbf', 'sigmoid'，分别代表线性核函数、多项式核函数、径向基核函数和Sigmoid核函数。下面逐一介绍它们的定义以及用法：
线性核函数（Linear Kernel）：线性核函数可以将样本点映射到更高维的空间中进行分类，其决策边界为线性超平面。线性核函数适合于样本特征维度较高，相对简单明了的情况。在SVC模型中，可以设置kernel='linear'启用线性核函数。
多项式核函数（Polynomial Kernel）：多项式核函数可以将样本点映射到更高维的空间中进行分类，其决策边界为多项式函数，可以对线性不可分的数据集进行有效分类。在SVC模型中，可以设置kernel='poly'启用多项式核函数，并通过degree参数来指定多项式核函数的度数，例如degree=2表示使用二次项函数。
径向基核函数（Radial Basis Function Kernel，RBF）：径向基核函数可以将样本点映射到更高维的空间中进行分类，其决策边界为以支持向量为中心的径向对称函数。径向基核函数是应用最为广泛的SVM核函数之一，在大部分情况下都可以取得不错的效果。在SVC模型中，可以设置kernel='rbf'启用径向基核函数，并通过gamma参数来调整决策边界的半径大小。gamma取值越小，决策边界的半径就越大，能够容许更多的错误分类点。
Sigmoid核函数：Sigmoid核函数可以将样本点映射到更高维的空间中进行分类，其决策边界为Sigmoid函数；与逻辑回归模型类似，Sigmoid核函数适合于处理二分类问题，但很少在SVM中使用。在SVC模型中，可以设置kernel='sigmoid'启用Sigmoid核函数。
'''