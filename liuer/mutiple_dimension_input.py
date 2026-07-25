import torch
import numpy as np

filename = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(filename[:, :-1]).float()
y_data = torch.from_numpy(filename[:, -1].reshape([-1, 1]))

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x



model = Model()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(1000):
    y_pre = model(x_data)
    loss = criterion(y_pre, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'epoch {epoch}, loss = {loss.item():.10f}')


print("\n模型参数：")
print("--- 第一层权重和偏置 (linear1) ---")
print(f'w = {model.linear1.weight.data},shape = {model.linear1.weight.data.shape}')
print(f'b = {model.linear1.bias.data}')

print("\n--- 第二层权重和偏置 (linear2) ---")
print(f'w = {model.linear2.weight.data},shape = {model.linear2.weight.data.shape}')
print(f'b = {model.linear2.bias.data}')

print("\n--- 第三层权重和偏置 (linear3) ---")
print(f'w = {model.linear3.weight.data},shape = {model.linear3.weight.data.shape}')
print(f'b = {model.linear3.bias.data}')

