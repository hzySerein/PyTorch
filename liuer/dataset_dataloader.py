import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


class MyDataset(Dataset):
    def __init__(self, filepath):
        data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.x_data = torch.from_numpy(data[:, :-1]).float()
        self.y_data = torch.from_numpy(data[:, -1].reshape([-1, 1]))

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


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


def main():
    dataset = MyDataset('diabetes.csv.gz')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    model = Model()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    for epoch in range(1000):
        total_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            y_pre = model(inputs)
            loss = criterion(y_pre, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'epoch {epoch}, avg_loss = {avg_loss:.10f}')


    print("\nlinear1 weight:")
    print(model.linear1.weight.data)
    print("\nlinear1 bias:")
    print(model.linear1.bias.data)

if __name__ == '__main__':
    main()
