# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle # 用于序列化和反序列化Python对象
from dataset.mnist import load_mnist # from dataset/mnist.py
from common.functions import sigmoid, softmax # from common/functions.py


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


''' init_network()会读入保存在pickle文件sample_weight.pkl中的学习到的
权重参数A。这个文件中以字典变量的形式保存了权重和偏置参数。
'''
def init_network():
    with open("row_code/ch03/sample_weight.pkl", 'rb') as f: # 'rb'表示以二进制格式读取
        network = pickle.load(f) # pickle.load()函数从文件中读取数据并返回
    return network

# predict()函数接收网络参数network和输入数据x，计算输出层的激活值y。
# 这里的网络参数包括权重矩阵W和偏置向量b。
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3'] # 获取权重参数
    b1, b2, b3 = network['b1'], network['b2'], network['b3'] # 获取偏置参数

    # 第一层
    a1 = np.dot(x, W1) + b1 # 计算第一层的加权和, dot()函数计算矩阵乘法
    z1 = sigmoid(a1) # 计算第一层的激活值
    # 第二层
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    # 输出层
    a3 = np.dot(z2, W3) + b3 # 计算输出层的加权和
    y = softmax(a3) # 计算输出层的激活值, softmax()函数将输出转换为概率分布

    return y


x, t = get_data() # 将测试集图像和标签赋值给x和t
network = init_network()
accuracy_cnt = 0 # 初始化正确计数器
for i in range(len(x)):
    y = predict(network, x[i]) # 对每个测试样本进行预测
    p= np.argmax(y) # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))