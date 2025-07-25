import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义函数 function_2
def function_2(x):
    return np.sum(x**2)

# 生成数据点
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# 计算函数值
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = function_2(np.array([X[i, j], Y[i, j]]))

# 创建 3D 图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制 3D 曲面图
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

# 添加颜色条和标签
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('function_2(x) = x² + y²')

# 设置视角
ax.view_init(elev=30, azim=35) # elev代表仰角，azim代表方位角

plt.show()