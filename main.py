import torch.optim as optim
import scipy.io as sio
import time
import torch
import h5py
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import cv2 as cv
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(5, 5))

        x = torch.randn(256, 256).view(-1, 1, 256, 256)
        self._to_linear = None
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


class CleanVsBlurred:
    CLEAN = 'PET'
    BLURRED = 'blurred'
    LABELS = {BLURRED: 0, CLEAN: 1}
    training_data = []
    BATCH_SIZE = 16
    EPOCHS = 5
    train_X = None
    train_y = None
    test_X = None
    test_y = None

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('GPU')
    else:
        device = torch.device("cpu")
        print('CPU')

    net = Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    def __init__(self, IS_Rebuild=0):
        if IS_Rebuild == 1:
            self.make_training_data()
        elif IS_Rebuild == 2:
            self.training_data = np.load("Ltraining_data.npy", allow_pickle=True)
        else:
            self.training_data = np.load("training_data.npy", allow_pickle=True)

        X = torch.Tensor([i[0] for i in self.training_data]).view(-1, 256, 256)
        X = X / 255.0
        y = torch.Tensor([i[1] for i in self.training_data])

        VAL_PCT = 0.1
        val_size = int(len(X) * VAL_PCT)
        print(val_size)

        self.train_X = X[:-val_size]
        self.train_y = y[:-val_size]

        self.test_X = X[-val_size:]
        self.test_y = y[-val_size:]

    def make_training_data(self):
        mat_content = sio.loadmat('data.mat')
        # mat_content = h5py.File('data.mat', 'r')
        print('Successfully Loaded:', mat_content.keys())

        for lbl in self.LABELS:
            print(lbl, mat_content[lbl].shape)
            ones_array = np.ones((256, 1))
            for i in tqdm(range(256)):
                img = mat_content[lbl][:, :, i]

                if sum(img.dot(ones_array)) > 0:
                    self.training_data.append([img, np.eye(2)[self.LABELS[lbl]]])

        print(len(self.training_data))
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)

    def fwd_pass(self, X, y, train=False):
        if train:
            self.net.zero_grad()
        outputs = self.net(X)

        matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]

        acc = matches.count(True) / len(matches)
        loss = self.loss_function(outputs, y)

        if train:
            loss.backward()
            self.optimizer.step()

        return int(acc * 100), int(loss * 100)

    def test(self):
        X, y = self.test_X, self.test_y
        val_acc, val_loss = self.fwd_pass(X.view(-1, 1, 256, 256).to(self.device), y.to(self.device))
        return val_acc, val_loss

    def train(self):
        MODEL_NAME = f"model-{int(time.time())}"
        print(MODEL_NAME)
        with open("model.log", "a") as f:
            for epoch in range(self.EPOCHS):
                for i in tqdm(range(0, len(self.train_X), self.BATCH_SIZE)):
                    batch_X = self.train_X[i:i + self.BATCH_SIZE].view(-1, 1, 256, 256)
                    batch_y = self.train_y[i:i + self.BATCH_SIZE]

                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    acc, loss = self.fwd_pass(batch_X, batch_y, train=True)

                    if i % 10 == 0:
                        val_acc, val_loss = self.test()
                        f.write(
                            f"{MODEL_NAME},{round(time.time(), 3)},{round(float(acc), 2)},{round(float(loss), 4)},"
                            f"{round(float(val_acc), 2)},{round(float(val_loss), 4)},{epoch}\n")


if __name__ == "__main__":
    clean_vs_blurred = CleanVsBlurred(1)
    clean_vs_blurred.train()
    clean_vs_blurred.confusion_matrix()
