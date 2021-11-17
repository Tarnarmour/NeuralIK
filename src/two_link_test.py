import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from robotics_src import Kinematics as kin
import csv
from NetworkModelFile import IKNet

n = 2
dh = [[0, 0, 1, 0]] * n
rev = ['r'] * n

arm = kin.SerialArm(dh, rev)


class IKDataset(Dataset):
    def __init__(self, train=True):
        if train:
            datapath = 'two_link_training_data.csv'
            labelpath = 'two_link_training_label.csv'
        else:
            datapath = 'two_link_test_data.csv'
            labelpath = 'two_link_test_label.csv'

        with open(datapath, mode='r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            data = []
            for row in csv_reader:
                if len(row) == 2:
                    data.append([])
                    data[-1].append(float(row[0]))
                    data[-1].append(float(row[1]))

        self.data = torch.tensor(data, dtype=torch.float32)

        with open(labelpath, mode='r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            label = []
            for row in csv_reader:
                if len(row) == 2:
                    label.append([])
                    label[-1].append(float(row[0]))
                    label[-1].append(float(row[1]))

        self.label = torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]

        return data, label


training_dataset = IKDataset(train=True)
test_dataset = IKDataset(train=False)

model = IKNet()

batch_size = 64
epoch_num = 15
learning_rate = 0.001

training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

loss_func = nn.MSELoss()
# loss_func = custom_loss
optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate, weight_decay=1e-3)


def train_loop(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    error = 0
    for i, (xs, qs) in enumerate(dataloader):
        x = qs
        y = xs
        yhat = model(x)
        loss = loss_func(yhat, y)
        error += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % int(size / batch_size / 10) == 0 and not i == 0:
            print(f"loss: {error / i:>4f} [{i * batch_size}/{size}]")


def test_loop(dataloader, model, loss_func, eps=0.001):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for xs, qs in dataloader:
            x = qs
            y = xs
            yhat = model(x)
            loss = loss_func(yhat, y).item()
            if loss < eps:
                correct += 1
            test_loss += loss
    test_loss /= size
    correct = correct / size * 100

    print(f"Average Loss: {test_loss:>3f} Percent Correct: {correct:>3f}%\n")


test_loop(test_dataloader, model, loss_func)

for epoch in range(epoch_num):
    print(f"Epoch {epoch + 1}\n----------------------------")
    train_loop(training_dataloader, model, loss_func, optimizer)
    test_loop(test_dataloader, model, loss_func)

# for param in model.parameters():
#     print(param)

torch.save(model.state_dict(), 'trained_model.pth')

print("Done!")
