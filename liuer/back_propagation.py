import torch

x_data = torch.tensor([1.0,2.0,3.0])
y_data = torch.tensor([2.0,4.0,6.0])

w = torch.tensor([1.0])
w.requires_grad = True

def forward(x,w):
    return w * x

def loss(x,y,w):
    y_pre = forward(x,w)
    return (y-y_pre)**2

for epoch in range(100):

    for x,y in zip(x_data,y_data):
        l = loss(x,y,w)
        l.backward()
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()

print('w = %.6f' % w.item())