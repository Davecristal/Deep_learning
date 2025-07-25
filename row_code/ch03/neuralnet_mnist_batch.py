import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("row_code/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

# 加入批处理,使用批处理来提高预测效率
batch_size = 100 # 批数量
accuracy_cnt = 0

for i in range(0, len(x), batch_size): # batch_size为每次处理的样本数量
    x_batch = x[i:i+batch_size] # 获取当前批次的样本
    y_batch = predict(network, x_batch) # 对当前批次的样本进行预测
    #矩阵的第0维是列方向,第1维是行方向
    p = np.argmax(y_batch, axis=1) # 获取每个样本预测的标签,axis=1表示沿着行的方向获取最大值的索引
    accuracy_cnt += np.sum(p == t[i:i+batch_size]) # 统计当前批次预测正确的样本数量

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
