import torch.optim as optim
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import cv2 as cv
import numpy as np

IS_Rebuild = 0


class CleanVsBlurred:
    CLEAN = 'PET'
    BLURRED = 'blurred'
    LABELS = {BLURRED: 0, CLEAN: 1}
    training_data = []

    def make_training_data(self):
        mat_content = sio.loadmat('data.mat')
        print(mat_content.keys())

        for lbl in self.LABELS:
            for i in tqdm(range(256)):
                img = mat_content[lbl][:, :, i]
                self.training_data.append([img, np.eye(2)[self.LABELS[lbl]]])
                cv.imshow(lbl, img)
                cv.waitKey(1)
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self._to_linear = 128 * 28 * 28

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        self.fc1 = nn.Linear(self._to_linear, 512)  # flattening.
        self.fc2 = nn.Linear(512, 2)  # 512 in, 2 out bc we're doing 2 classes (clean vs blurred).

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)


def train(net, train_X, train_y):
    optimizer = optim.Adam(net.parameters(), lr=0.0008)
    loss_function = nn.MSELoss()
    BATCH_SIZE = 16
    EPOCHS = 10

    for epoch in range(EPOCHS):
        loss = 0
        for i in tqdm(range(0, len(train_X),
                            BATCH_SIZE)):  # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
            # print(f"{i}:{i+BATCH_SIZE}")
            batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 256, 256)
            batch_y = train_y[i:i + BATCH_SIZE]

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            net.zero_grad()

            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()  # Does the update

        print(f"Epoch: {epoch}. Loss: {loss}")


def test(net, test_X, test_y):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            print(test_y[i])
            real_class = torch.argmax(test_y[i])

            net_out = net(test_X[i].view(-1, 1, 256, 256).to(device))[0]  # returns a list,
            predicted_class = torch.argmax(net_out).to(device)

            print(net_out)

            """img = test_X[i] * 255
            cv.imshow(f'{i}: {predicted_class}, {real_class}', img.view(256, 256).cpu().detach().numpy())
            cv.waitKey(0)
            cv.destroyWindow(f'{i}: {predicted_class}, {real_class}')"""

            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy: ", round(correct / total, 3))


if __name__ == "__main__":
    clean_vs_blurred = CleanVsBlurred()
    if IS_Rebuild:
        clean_vs_blurred.make_training_data()
    else:
        clean_vs_blurred.training_data = np.load("training_data.npy", allow_pickle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('GPU')
    else:
        device = torch.device("cpu")
        print('CPU')

    net = Net().to(device)

    X = torch.Tensor([i[0] for i in clean_vs_blurred.training_data]).view(-1, 256, 256)
    X = X / 255.0
    y = torch.Tensor([i[1] for i in clean_vs_blurred.training_data])

    VAL_PCT = 0.1  # lets reserve 10% of our data for validation
    val_size = int(len(X) * VAL_PCT)
    print(val_size)

    train_X = X[:-val_size]
    train_y = y[:-val_size]
    train(net, train_X, train_y)

    test_X = X[-val_size:]
    test_y = y[-val_size:]

    test_X = test_X.to(device)
    test_y = test_y.to(device)
    test(net, test_X, test_y)
