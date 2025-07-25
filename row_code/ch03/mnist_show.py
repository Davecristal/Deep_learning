# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image # Pillow库用于处理图像


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # 将NumPy数组转换为PIL图像对象
    pil_img.show()

# 载入MNIST数据集
# flatten=True表示将图像展平为一维数组，normalize=False表示不进行归一化处理
# load_mnist函数返回四个值：训练集图像、训练集标签、测试集图像、测试集标签
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[100] # 获取第n张图像
label = t_train[100] # 获取第n张图像对应的标签
print(label)

print(img.shape)
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
print(img.shape)

img_show(img)
