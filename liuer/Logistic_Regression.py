# logistic regression
# 逻辑回归模型
# 用于分类问题
# sigmoid函数
# 损失函数：交叉熵损失 loss = -(y*log(y_pre)+(1-y)*log(1-y_pre))
# 交叉熵特点：当y_pre与y相同时，损失为0；当y_pre与y不同时，损失为1。
import torch
import torch.nn.functional as F



x_data= torch.tensor([[1.0],[2.0],[3.0]])
y_data= torch.tensor([[0.0],[0.0],[1.0]])

class LogisticRegression(torch.nn.Module):
        def __init__(self):
                super().__init__()
                # 线性层
                self.linear = torch.nn.Linear(1, 1)

        def forward(self, x):
                y_pre = F.sigmoid(self.linear(x))
                return y_pre



model = LogisticRegression()
epochs = range(1000)
# 损失函数
criterion = torch.nn.BCELoss()
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in epochs:
        y_pre = model(x_data)
        loss = criterion(y_pre, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch: %d loss: %.10f' % (epoch, loss.item()))

# 打印模型参数
print('w: %.10f' % (model.linear.weight.item()))
print('b: %.10f' % (model.linear.bias.item()))

# 测试
x_test = torch.tensor([[4.0]])
y_pre = model(x_test)
print('y_pre: %.10f' % y_pre.item())








