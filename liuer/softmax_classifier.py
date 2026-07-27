# 多分类问题
# 激活函数为softmax函数（softmax(x) = exp(x) / sum(exp(x))）
# 损失函数为交叉熵损失函数（CrossEntropyLoss loss = -sum(y * log(softmax(x)))）
# pytorch里面的CrossEntropyLoss函数已经包含了softmax函数，所以不需要手动实现softmax函数

import torch

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data/mnist/',
    train=True,
    download=True,
    transform=transform)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=False)

test_dataset = datasets.MNIST(
    root='./data/mnist/',
    train=False,
    download=True,
    transform=transform)

test_loader = DataLoader(test_dataset,
                          batch_size=batch_size,
                          shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.5)
criterion = torch.nn.CrossEntropyLoss()


def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader):

        inputs,labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % 100 == 99:
            print(f'epoch {epoch}, batch_idx {batch_idx}, loss = {running_loss:.10f}')
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs,labels = data
            outputs = model(inputs)
            _,predicted = torch.max(outputs.data,dim = 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'acc = {correct / total: .10f}')

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
