import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0,4.0]
y_data = [2.0,4.0,6.0,8.0]

def  forward(w,x):
    return x * w

def loss(x,y,w):
    y_pre = forward(w,x)
    return (y - y_pre)**2

w_list = []
mse_list = []

for w in np.arange(0,4.1,0.1):
    print('w = ',w)
    l_sum = 0
    for x,y in zip(x_data,y_data):
        y_pre = forward(x,w)
        loss_val = loss(x,y,w)
        l_sum += loss_val
        print(x,y,y_pre,l_sum)


    w_list.append(w)
    mse_list.append(l_sum/4)
    print(l_sum/4)

# 绘图：
plt.plot(w_list, mse_list, label='MSE')
plt.ylabel('mse')
plt.xlabel('w')
plt.legend()
plt.show()



