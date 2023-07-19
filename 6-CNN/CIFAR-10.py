from keras.datasets import cifar10
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from matplotlib import pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

'''一、数据的导入'''
np.random.seed(10)
# 50000训练集,10000测试集
# x_train:50000*32*32 y_train:50000 x_test:10000*32*32 y_test:10000
(x_img_train, y_label_train), (x_img_test, y_label_test) = cifar10.load_data()
# print(x_img_train[0,:,:,0]) # 某个图像的某通道
# print(y_label_train) # 所有标签

'''二、数据的处理'''
# 标准化
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0
# One-Hot编码
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
# print(y_label_train_OneHot)

'''三、建立模型'''
# 创建多层顺序连接的神经网络
model = Sequential()
# 卷积层 1，32*32的图像，共32个
model.add(Conv2D(filters=32, kernel_size=(3, 3),  # 32个3*3的卷积核
                 input_shape=(32, 32, 3),  # 形状:32高 * 32宽 * 3通道
                 activation='relu',  # Relu激活函数
                 padding='same'))  # 输入输出尺寸相同
# Dropout层，随机丢弃25%输入神经元（置为0）
model.add(Dropout(0.25))
# 池化层（降采样层） 1，16*16的图像，共32个
model.add(MaxPooling2D(pool_size=(2, 2)))
# 卷积层 2，16*16的图像，共64个
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))  # 64个3*3的卷积核
model.add(Dropout(0.25))
# 池化层（降采样层） 2，8*8的图像，共64个
model.add(MaxPooling2D(pool_size=(2, 2)))
# 平坦层 8*8*64个神经元
model.add(Flatten())
model.add(Dropout(rate=0.25))
# 隐藏层（全连接层） 1024个神经元
model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.25))
# 输出层（全连接层） 对应0-9这10个类别
model.add(Dense(10, activation='softmax'))

'''四、训练模型'''
# 编译模型（误差函数交叉熵、Adam梯度下降、指标准确度）
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 训练模型
train_history = model.fit(x_img_train_normalize, y_label_train_OneHot,  # 训练集
                          validation_split=0.2,  # 20%用作验证集
                          epochs=15, batch_size=128, verbose=1)  # 10次迭代训练、每批次 128张，输出记录
# print(train_history.history)

'''五、测试模型'''
scores = model.evaluate(x_img_test_normalize, y_label_test_OneHot, verbose=0)
# print(scores)

'''六、相关信息可视化'''


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train,', 'validation'], loc='upper left')


# 1.准确率变化曲线
plt.figure(1)
show_train_history(train_history, 'accuracy', 'val_accuracy')
# 2.损失率变化曲线
plt.figure(2)
show_train_history(train_history, 'loss', 'val_loss')
# 3.输出25张原数据集的图像
label_dict = {0: '飞机', 1: '汽车', 2: '鸟', 3: '猫', 4: '鹿', 5: "狗", 6: '青蛙', 7: '马', 8: '船', 9: '卡车'}


# 显示几张图片和标签
def show_images_labels_prediction(images, labels, prediction, idx, num=10):
    flig = plt.figure(figsize=(12, 14))
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')
        title = '标签：' + str(label_dict[labels[i][0]])
        if len(prediction) > 0:
            title += ',预测：' + label_dict[prediction[i]]
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1


show_images_labels_prediction(x_img_train, y_label_train, [], 0, 25)
# 4.显示测试集中预测和真实标签
predicted_probability = model.predict(x_img_test_normalize)
prediction = np.argmax(predicted_probability, axis=-1)
# print(prediction)
show_images_labels_prediction(x_img_test, y_label_test, prediction, 0, 25)
# 5.混淆矩阵
confusion_matrix = pd.crosstab(y_label_test.reshape(-1), prediction, rownames=['label'], colnames=['predict'])
print(confusion_matrix)
# confusion_matrix.head()
# 6.查看完整神经网络的构架层次
model.summary()
# 7.准确率
print("准确率：{:.4f}%".format(scores[1] * 100))

plt.show()
