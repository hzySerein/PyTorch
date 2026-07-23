"""
import matplotlib.pyplot as plt

x_data = [1,2,3,4]
y_data = [2,4,6,8]
w = 4
def forward(w,x):
    return w*x

def mse(xs,ys):
    loss = 0
    for x,y in zip(xs,ys):
        y_pre_val = forward(w,x)
        loss += (y-y_pre_val)**2
    return loss/len(xs)

def gradient(xs,ys):
    grad = 0
    for x,y in zip(xs,ys):
        grad += 2*x*(x*w-y)

    return grad/len(xs)

w_list = []
mse_list = []

epochs = range(100)
for epoch in epochs:
    grad = gradient(x_data,y_data)
    mse_val = mse(x_data,y_data)
    w -= grad*0.01
    w_list.append(w)
    mse_list.append(mse_val)

plt.plot(epochs,mse_list,color = 'r',label='mse')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.legend()
plt.grid()
plt.show()

"""

import matplotlib.pyplot as plt
import numpy as np

x_data = [1,2,3,4]
y_data = [2,4,6,8]

def forward(x,w):
    return w * x

def mse(xs,ys,w):
    l_sum = 0
    for x ,y in zip(xs,ys):
        y_pre = forward(x,w)
        loss = (y-y_pre)**2
        l_sum += loss

    return l_sum/(len(xs))

def gradient(xs,ys,w):
    grad = 0
    for x,y in zip(xs,ys):
        grad += 2 * x * (w * x - y)

    return  grad / len(xs)

w = 5
w_list = []
for epoch in range(100):
    mse_val = mse(x_data,y_data,w)
    grad_val = gradient(x_data,y_data,w)
    w = w - 0.01 * grad_val
    w_list.append(w)
    print('epoch = ',epoch,'w = ', w,'mse = ',mse_val)

print(w)


plt.plot(range(100),w_list,label = 'w')
plt.xlabel('epoch')
plt.ylabel('w')
plt.legend()
plt.show()
