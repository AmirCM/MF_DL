import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
import torch.optim as optim
import random

CLEAN = 'PET'
BLURRED = 'blurred'
LABELS = {BLURRED: 0, CLEAN: 1}


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3))

        x = torch.randn(256, 256).view(-1, 1, 256, 256)
        self._to_linear = 115200
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def test(batch_X, batch_y, net, device):
    with torch.no_grad():
        loss_function = nn.MSELoss()
        batch_X = batch_X.view(-1, 1, 256, 256).to(device)
        batch_y = batch_y.to(device)
        batch_out = net(batch_X)
        val_loss = loss_function(batch_out, batch_y)
        matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(batch_out, batch_y)]
        val_acc = matches.count(True) / len(matches)
        return val_acc, val_loss


def train():
    MODEL_NAME = f"model-{int(time.time())}"
    print(f'Model tag: {MODEL_NAME}')
    device = 'cuda:0'
    net = Net().to("cuda:0")

    training_data = np.load("training_data.npy", allow_pickle=True)
    X = torch.Tensor([i[0] for i in training_data]).view(-1, 256, 256)
    y = torch.Tensor([i[1] for i in training_data])

    VAL_PCT = 0.1
    val_size = int(len(X) * VAL_PCT)
    print(f'Test size: {val_size}')

    train_X = X[:-val_size]
    train_y = y[:-val_size]

    test_X = X[-val_size:]
    test_y = y[-val_size:]

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    BATCH_SIZE = 8
    EPOCHS = 5

    for epoch in range(EPOCHS):
        loss = 0
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 256, 256)
            batch_y = train_y[i:i + BATCH_SIZE]

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            net.zero_grad()
            outputs = net(batch_X)

            loss = loss_function(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()  # Gradient Descent

            with open("model.log", "a") as f:
                matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, batch_y)]
                acc = matches.count(True) / len(matches)
                random_st = random.randint(0, 35 - BATCH_SIZE)
                val_acc, val_loss = test(test_X[random_st:random_st + BATCH_SIZE],
                                         test_y[random_st:random_st + BATCH_SIZE], net, device)
                f.write(
                    f"{MODEL_NAME},{round(time.time(), 3)},{round(float(acc), 2)},{round(float(loss), 4)},"
                    f"{round(float(val_acc), 2)},{round(float(val_loss), 4)},{epoch}\n")

        print(f"Epoch: {epoch}. Loss: {round(float(loss*100), 3)}")

    with torch.no_grad():
        correct_clean = 0
        wrong_clean = 0
        correct_blurred = 0
        wrong_blurred = 0

        for i in range(0, len(test_X), BATCH_SIZE):
            batch_X = test_X[i:i + BATCH_SIZE].view(-1, 1, 256, 256).to(device)
            batch_y = test_y[i:i + BATCH_SIZE].to(device)
            batch_out = net(batch_X)

            matches = []
            for k, j in zip(batch_out, batch_y):
                matches += [torch.argmax(k) == torch.argmax(j)]
                if torch.argmax(k) == torch.argmax(j):
                    if torch.argmax(j) == LABELS[BLURRED]:
                        correct_blurred += 1
                    else:
                        correct_clean += 1
                else:
                    if torch.argmax(j) == LABELS[BLURRED]:
                        wrong_blurred += 1
                    else:
                        wrong_clean += 1
            acc = matches.count(True) / len(matches)

            print(f'Batch Accuracy {i}: {round(acc, 3)}')

    overall_acc = (correct_clean + correct_blurred) / (correct_clean + correct_blurred + wrong_clean + wrong_blurred)
    print(f'Overall accuracy: {round(overall_acc, 3)}')
    print('\tClean\tBlurred')
    print(f'Clean\t{correct_clean}\t{wrong_clean}')
    print(f'Blurred\t{wrong_blurred}\t{correct_blurred}')
    with open("confusion_matrix.log", "a") as f:
        log_tag = f"ResNet-{int(time.time())}"
        print(f'File tag: {log_tag}')
        f.write(f'{log_tag},{correct_clean},{wrong_clean},{correct_blurred},{wrong_blurred}')


train()
torch.cuda.empty_cache()
