import numpy as np
import matplotlib.pyplot as plt

# 数据
x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 5.0, 7.0]

def forward(x,w,b):
    return w * x + b

def loss(x,y,w,b):
    y_pre_val = forward(x,w,b)
    return (y_pre_val - y)**2

w_list = np.arange(0.0, 4.1, 0.1)
b_list = np.arange(-2.0, 2.1, 0.1)


mse_list = []
for w in w_list:
    for b in b_list:
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            loss_val = loss(x_val, y_val,w,b)
            l_sum += loss_val
        mse_list.append(l_sum / 3)


# 3D绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
W, B = np.meshgrid(w_list, b_list)
MSE = np.array(mse_list).reshape(len(b_list), len(w_list))

ax.plot_surface(W, B, MSE)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
plt.show()