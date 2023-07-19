from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

'''
sklearn.model_selection是scikit-learn中的交叉验证模块，
提供了许多用于模型选择和参数调优的工具。
交叉验证是一种评估统计模型精度的方法。
它的基本思想是重复地使用数据，将给定的数据集划分为两部分，
一部分用于模型的训练，另一部分用于模型的评估。
model_selection中的一些常用函数包括：
train_test_split()：随机划分数据集为训练集和测试集。默认情况下，训练集占总数据集的75%，测试集占25%。
KFold()：k折交叉验证生成器，可以用于将数据集划分为k个连续的、非重叠的折叠子集。
StratifiedKFold()：分层k折交叉验证生成器，与KFold()相似，但是它会在每个折叠中保持各个类别的比例。
cross_val_score()：基于给定模型对数据进行k折交叉验证，并返回每个折叠的测试得分。
GridSearchCV()：用于通过交叉验证来进行参数调优的函数。给定各种参数和取值范围，以及评分准则，该函数可以通过网格搜索来找到最佳的模型参数组合。
使用model_selection中的这些函数可以帮助我们更轻松地进行交叉验证及模型的参数调优
'''

iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
'''
train_test_split()函数是Scikit-Learn中的一个用于将数据集划分为训练集和测试集的函数。
除了必须输入的数据集（或特征矩阵）和相应的标签向量之外，它还有一些可选参数。下面列出了一些常用的参数：
test_size：指定测试集占总数据集的比例，默认值为0.25；
train_size：指定训练集占总数据集的比例（互补参数，如果指定就不能同时指定test_size），默认值为0.75；
random_state：指定生成随机数的种子值，保证每次生成的随机数一致（即使多次运行也是一样的）；
shuffle：是否在分割数据之前对数据进行随机排序（默认True），如果为False，则会按顺序分配；
stratify：指定按照什么方式划分数据，一般指定为类似标签的一维数组，以保证划分结果中每个类的比例大致相等。
'''
