import numpy as np

# 定义均值向量和协方差矩阵
mu_1 = np.array([-2, -2])
sigma_1 = np.array([[1, 0], [0, 1]])
mu_2 = np.array([2, 2])
sigma_2 = np.array([[1, 0], [0, 4]])

# 生成数据
train_1 = np.random.multivariate_normal(mu_1, sigma_1, 500)
test_1 = np.random.multivariate_normal(mu_1, sigma_1, 100)
train_2 = np.random.multivariate_normal(mu_2, sigma_2, 500)
test_2 = np.random.multivariate_normal(mu_2, sigma_2, 100)

# 添加标签
train_data = np.vstack((train_1, train_2))
test_data = np.vstack((test_1, test_2))
train_labels = np.hstack((np.zeros(500), np.ones(500)))
test_labels = np.hstack((np.zeros(100), np.ones(100)))

# 打乱数据
train_idx = np.random.permutation(len(train_data))
test_idx = np.random.permutation(len(test_data))
train_data = train_data[train_idx]
train_labels = train_labels[train_idx]
test_data = test_data[test_idx]
test_labels = test_labels[test_idx]

import matplotlib.pyplot as plt

# 绘制训练集
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels)
plt.title('Training Data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# 绘制测试集
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels)
plt.title('Test Data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()


def nearest_neighbor(train_data, train_labels, test_data):
    n_train = len(train_data)
    n_test = len(test_data)
    pred_labels = np.zeros(n_test)

    for i in range(n_test):
        test = test_data[i]
        min_dist = np.Inf

        for j in range(n_train):
            dist = np.linalg.norm(test - train_data[j])
            if dist < min_dist:
                min_dist = dist
                nearest = train_labels[j]

        pred_labels[i] = nearest

    return pred_labels

# 对测试集进行预测
pred_labels = nearest_neighbor(train_data, train_labels, test_data)

# 计算平均错误率
accuracy = np.mean(pred_labels == test_labels)
print('Accuracy of Nearest-Neighbor Classifier: {:.2f}%'.format(accuracy * 100))

def rbf_kernel(x, y, gamma):
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)

def smo(train_data, train_labels, C, toler, gamma, max_iter):
    n_train = len(train_data)
    alphas = np.zeros(n_train)
    b = 0
    iter = 0

    while iter < max_iter:
        alpha_pairs_changed = 0

        for i in range(n_train):
            f_i = np.sum(alphas * train_labels * rbf_kernel(train_data, train_data[i], gamma)) + b
            E_i = f_i - train_labels[i]

            if (train_labels[i] * E_i < -toler and alphas[i] < C) or (train_labels[i] * E_i > toler and alphas[i] > 0):
                j = np.random.choice([k for k in range(n_train) if k != i])
                f_j = np.sum(alphas * train_labels * rbf_kernel(train_data, train_data[j], gamma)) + b
                E_j = f_j - train_labels[j]

                alpha_i_old, alpha_j_old = alphas[i], alphas[j]

                if train_labels[i] != train_labels[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])

                if L == H:
                    continue

                eta = 2 * rbf_kernel(train_data[i], train_data[j], gamma) - rbf_kernel(train_data[i], train_data[i], gamma) - rbf_kernel(train_data[j], train_data[j], gamma)
                if eta >= 0:
                    continue

                alphas[j] -= train_labels[j] * (E_i - E_j) / eta
                alphas[j] = np.clip(alphas[j], L, H)

                if abs(alphas[j] - alpha_j_old) < 1e-5:
                    continue

                alphas[i] += train_labels[i] * train_labels[j] * (alpha_j_old - alphas[j])
                b1 = b - E_i - train_labels[i] * (alphas[i] - alpha_i_old) * rbf_kernel(train_data[i], train_data[i], gamma) - train_labels[j] * (alphas[j] - alpha_j_old) * rbf_kernel(train_data[i], train_data[j], gamma)
                b2 = b - E_j - train_labels[i] * (alphas[i] - alpha_i_old) * rbf_kernel(train_data[i], train_data[j], gamma) - train_labels[j] * (alphas[j] - alpha_j_old) * rbf_kernel(train_data[j], train_data[j], gamma)

                if alphas[i] > 0 and alphas[i] < C:
                    b = b1
                elif alphas[j] > 0 and alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                alpha_pairs_changed += 1

        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0

    return alphas, b

def svm(train_data, train_labels, test_data, C, toler, gamma, max_iter):
    alphas, b = smo(train_data, train_labels, C, toler, gamma, max_iter)
    n_train = len(train_data)
    n_test = len(test_data)
    pred_labels = np.zeros(n_test)

    for i in range(n_test):
        f_i = np.sum(alphas * train_labels * rbf_kernel(train_data, test_data[i], gamma)) + b
        pred_labels[i] = 1 if f_i > 0 else 0

    return pred_labels

# 对测试集进行预测
pred_labels = svm(train_data, train_labels, test_data, C=1, toler=1e-3, gamma=0.5, max_iter=1000)

# 计算平均错误率
accuracy = np.mean(pred_labels == test_labels)
print('Accuracy of SVM Classifier: {:.2f}%'.format(accuracy * 100))



import numpy as np

# 生成第一类样本数据
m1 = np.array([-2, -2])
cov1 = np.array([[1, 0], [0, 1]])
train_data1 = np.random.multivariate_normal(m1, cov1, 500)
test_data1 = np.random.multivariate_normal(m1, cov1, 100)

# 生成第二类样本数据
m2 = np.array([2, 2])
cov2 = np.array([[1, 0], [0, 4]])
train_data2 = np.random.multivariate_normal(m2, cov2, 500)
test_data2 = np.random.multivariate_normal(m2, cov2, 100)

# 合并样本数据
train_data = np.vstack((train_data1, train_data2))
test_data = np.vstack((test_data1, test_data2))

# 生成标签数据，第一类标签为-1，第二类标签为1
train_label = np.hstack((-np.ones(500), np.ones(500)))
test_label = np.hstack((-np.ones(100), np.ones(100)))

import matplotlib.pyplot as plt

# 绘制训练集散点图
plt.figure(figsize=(8, 6))
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_label, cmap="coolwarm")
plt.title("Training Set")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# 绘制测试集散点图
plt.figure(figsize=(8, 6))
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label, cmap="coolwarm")
plt.title("Test Set")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# 定义SMO算法
def SVM_SMO_train(X, Y, C, max_iter, tol):
    # 初始化
    m, n = X.shape
    alpha = np.zeros(m)
    b = 0
    E = np.zeros(m)
    K = np.dot(X, X.T) * Y.reshape(m, 1) * Y.reshape(1, m)

    # 开始迭代
    iter_count = 0
    while iter_count < max_iter:
        alpha_pair_changed = 0
        for i in range(m):
            E[i] = np.sum(alpha * Y * K[i]) + b - Y[i]
            if (Y[i] * E[i] < -tol and alpha[i] < C) or (Y[i] * E[i] > tol and alpha[i] > 0):
                j = np.random.choice(np.delete(np.arange(m), i))
                eta = K[i, i] + K[j, j] - 2 * K[i, j]
                if eta <= 0:
                    continue
                alpha_j_new = alpha[j] + Y[j] * (E[i] - E[j]) / eta
                if Y[i] == Y[j]:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                else:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                if L == H:
                    continue
                alpha[i], alpha[j] = clip_alpha(alpha[i], alpha[j], L, H)
                b1 = -E[i] - Y[i] * K[i, i] * (alpha[i] - alpha[i])
                b2 = -E[j] - Y[j] * K[j, i] * (alpha[j] - alpha[j])
                if alpha[i] > 0 and alpha[i] < C:
                    b = b1
                elif alpha[j] > 0 and alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                alpha_pair_changed += 1
        if alpha_pair_changed == 0:
            iter_count += 1
        else:
            iter_count = 0
    return alpha, b


# 定义预测函数
def SVM_predict(X, weight, b):
    y_pred = np.dot(X, weight) + b
    y_pred = np.sign(y_pred)
    return y_pred


# 定义计算错误率函数
def error_rate(y_pred, y_true):
    return np.sum(y_pred != y_true) / len(y_true)


# 将alpha调整为满足0 <= alpha <= C的值
def clip_alpha(alpha, alpha_i, L, H):
    if alpha_i > H:
        alpha_i = H
    if alpha_i < L:
        alpha_i = L
    alpha_j = alpha - alpha_i
    if alpha_j > 0:
        alpha_j = H
    else:
        alpha_j = L
    alpha_i = alpha_i + alpha_j
    alpha_j = alpha_j - alpha_j
    return alpha_i, alpha_j


# 训练SVM模型
alpha, b = SVM_SMO_train(train_data, train_label, 1, 100, 1e-3)
weight = np.dot(train_data.T, alpha * train_label)

# 使用SVM模型预测测试集的类别
y_pred = SVM_predict(test_data, weight, b)
print("错误率为：", error_rate(y_pred, test_label))

