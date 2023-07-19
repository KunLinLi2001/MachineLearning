import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import np_utils

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

'''一、数据的导入'''
np.random.seed(10)
(x_img_train, y_label_train), (x_img_test, y_label_test) = keras.datasets.mnist.load_data()

'''二、数据的处理'''
# 标准化
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0
# 数据平铺
x_img_train_reshape = x_img_train_normalize.reshape(-1, 784)
x_img_test_reshape = x_img_test_normalize.reshape(-1, 784)
# One-Hot编码
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)

'''三、建立模型'''
model = keras.Sequential([
    # 隐含层 1 -- 64结点
    keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    # 隐含层 2 -- 32结点
    keras.layers.Dense(32, activation='relu'),
    # 隐含层 3 -- 16结点
    keras.layers.Dense(16, activation='relu'),
    # 输出层（全连接层） 对应0-9这10个数字
    keras.layers.Dense(10, activation='softmax')
])

'''四、训练模型'''
# 编译模型（误差函数交叉熵、Adam梯度下降、指标准确度）
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# 训练模型
train_history = model.fit(x_img_train_reshape,y_label_train,
                          validation_split=0.2,  # 20%用作验证集
                          epochs=10, batch_size=32, verbose=1)

'''五、测试模型'''
scores = model.evaluate(x_img_test_reshape, y_label_test, verbose=0)

'''六、相关信息可视化'''
# 可视化历史记录
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train,', 'validation'], loc='upper left')
# 显示几张图片和标签
def show_images_labels_prediction(images, labels, prediction, idx, num=10):
    flig = plt.figure(figsize=(12, 14))
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')
        title = '标签：' + str(label_dict[labels[i]])
        if len(prediction) > 0:
            title += ',预测：' + label_dict[prediction[i]]
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1

# 1.准确率变化曲线
plt.figure(1)
show_train_history(train_history, 'accuracy', 'val_accuracy')
# 2.损失率变化曲线
plt.figure(2)
show_train_history(train_history, 'loss', 'val_loss')
# 3.输出25张原数据集的图像
label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: "5", 6: '6', 7: '7', 8: '8', 9: '9'}
show_images_labels_prediction(x_img_train, y_label_train, [], 0, 25)
# 4.显示测试集中预测和真实标签
predicted_probability = model.predict(x_img_test_reshape)
prediction = np.argmax(predicted_probability, axis=-1)
show_images_labels_prediction(x_img_test, y_label_test, prediction, 0, 25)
# 5.混淆矩阵
confusion_matrix = pd.crosstab(y_label_test.reshape(-1), prediction, rownames=['label'], colnames=['predict'])
print(confusion_matrix)
# 6.查看完整神经网络的构架层次
model.summary()
# 7.准确率
print("准确率：{:.4f}%".format(scores[1] * 100))

plt.show()
