import torch

x_data = torch.tensor([[1.0],[2.0],[3.0]])
y_data = torch.tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        return self.linear(x)

model = LinearModel()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

for epoch in range(5000):
    y_pre = model(x_data)
    loss = criterion(y_pre,y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch = %d loss = %.10f' %(epoch ,loss.item()))


print('w = %.10f b = %.10f' %(model.linear.weight.item(),model.linear.bias.item()))

# Test
x_test = torch.tensor([[4.0]])
y_pre = model(x_test)
print('y_pre: %.10f' % y_pre.item())