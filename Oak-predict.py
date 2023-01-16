# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
from time import *
import GLCM
from scipy import linalg

# 數據集加载函数，統一處理imgheight*imgwidth的大小，同時設置batch
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # 加载訓練集
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # 加载測試集
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names

    return train_ds, val_ds, class_names



def model_load(IMG_SHAPE=(224, 224, 3), class_num=5):
    # 搭建模型
    model = tf.keras.models.Sequential([
        # 對模型進行歸一化的處理，也就是將0-255之間的樹變成0-1之間
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
        # 卷積層，該卷積層的輸出為32個通道，卷積核的大小是3*3，激活函數為relu
        tf.keras.layers.Conv2D(32, (2,2), activation='relu'),
        # 卷積層，輸出為64個通道，卷積核大小為3*3，激活函數為relu
        tf.keras.layers.MaxPooling2D(2,2),
        # 添加池化層，池化的kernel大小是2*2
        tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
        # 卷積層，輸出1284個通道，卷積核大小為3*3，激活函數為relu
        tf.keras.layers.MaxPooling2D(2, 2),
        # 添加池化層，池化的kernel大小是2*2
        tf.keras.layers.Conv2D(128, (2,2), activation='relu'),
        # 卷積層，輸出為256個通道，卷積核大小為3*3，激活函數為relu
        tf.keras.layers.MaxPooling2D(2, 2),
        # 添加池化層，池化的kernel大小是2*2
        tf.keras.layers.Flatten(),
        # 將二維的輸出轉化為一維
        tf.keras.layers.Dense(4096, activation='relu'),
        # 通過softmax函數將模型輸出為類名長度的神經元上，激活函數採用softmax對應概率值
        tf.keras.layers.Dense(128, activation='relu'),
        # 通過softmax函數將模型輸出為類名長度的神經元上，激活函數採用softmax對應概率值
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    # 输出模型信息
    model.summary()
    # 指明模型的訓練參數，優化器為sgd優化器，損失函數為交叉熵損失函數
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    # 返回模型
    return model


# 展示訓練過程的曲線
def show_loss_acc(history):
    # 從history中提取模型訓練集和驗證集準確率信息和誤差信息
    max_val_acc=0
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('./results/Mean_cnn.png', dpi=100)
    
    for i in val_acc:
        if(i>max_val_acc):
            max_val_acc=i
    print('Max val_accuracy = '+str(i))

def train(epochs):
    # 開始訓練，記錄開始時間
    begin_time = time()
    # 加載數據集
    train_ds, val_ds, class_names = data_load("../GLCM_Test_picture/Mean/train",
                                              "../GLCM_Test_picture/Mean/test", 224, 224, 8)
    print(class_names)
    # 加载模型
    model = model_load(class_num=len(class_names))
    # 指明訓練的輪數epoch，開始訓練
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    # 保存模型
    model.save("./GLCM+CNN_model/Mean_cnn.h5")
    # 記錄結束時間
    end_time = time()
    run_time = end_time - begin_time
    print('該循環程序運行時間：', run_time, "s") 
    # 繪製模型訓練過程圖
    show_loss_acc(history)


if __name__ == '__main__':
    train(epochs=40)