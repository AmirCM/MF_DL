import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import numpy as np
from tqdm import tqdm
import time
import torch.optim as optim

CLEAN = 'PET'
BLURRED = 'blurred'
LABELS = {BLURRED: 0, CLEAN: 1}
VGG16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
torch.cuda.empty_cache()

class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG16)

        self.fcs = nn.Sequential(
            nn.Linear(65536, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


if not torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('GPU')
else:
    device = torch.device("cpu")
    print('CPU')

net = VGG_net(in_channels=1, num_classes=2).to(device)

training_data = np.load("training_data.npy", allow_pickle=True)
X = torch.Tensor([i[0] for i in training_data]).view(-1, 256, 256)
X = X / 255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X) * VAL_PCT)
print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

BATCH_SIZE = 16
EPOCHS = 1
num_clean = 0
num_blur = 0
for epoch in range(EPOCHS):
    loss = 0
    for i in range(0, len(train_X), BATCH_SIZE):
        batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 256, 256)
        batch_y = train_y[i:i + BATCH_SIZE]

        for y in batch_y:
            if torch.argmax(y) == LABELS[BLURRED]:
                num_blur += 1
            else:
                num_clean += 1

        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        net.zero_grad()
        outputs = net(batch_X)

        loss = loss_function(outputs, batch_y)
        print(f'Loss: {loss}')
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()  # Gradient Descent

    print(f"Epoch: {epoch}. Loss: {loss}")

print(num_clean, num_blur)
correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        batch_X = test_X[i:i + BATCH_SIZE].view(-1, 1, 256, 256).to(device)
        batch_y = test_y[i:i + BATCH_SIZE].to(device)
        batch_out = net(batch_X)

        out_maxes = [torch.argmax(i) for i in batch_out]
        target_maxes = [torch.argmax(i) for i in batch_y]
        for i, j in zip(out_maxes, target_maxes):
            if i == j:
                correct += 1
            total += 1

print("Accuracy: ", round(correct / total, 3))

X, y = test_X.view(-1, 1, 256, 256).to(device), test_y.to(device)
outputs = net(X)
correct_clean = 0
wrong_clean = 0
correct_blurred = 0
wrong_blurred = 0
for i, j in zip(outputs, y):
    if torch.argmax(i) == torch.argmax(j):
        if torch.argmax(j) == LABELS[BLURRED]:
            correct_blurred += 1
        else:
            correct_clean += 1
    else:
        if torch.argmax(j) == LABELS[BLURRED]:
            wrong_blurred += 1
        else:
            wrong_clean += 1

print('\tClean\tBlurred')
print(f'Clean\t{correct_clean}\t{wrong_clean}')
print(f'Blurred\t{wrong_blurred}\t{correct_blurred}')
