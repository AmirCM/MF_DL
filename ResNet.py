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


class block(nn.Module):
    def __init__(
            self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


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
    net = ResNet101(img_channel=1, num_classes=2).to("cuda:0")

    training_data = np.load("training_data.npy", allow_pickle=True)
    X = torch.Tensor([i[0] for i in training_data]).view(-1, 256, 256)
    X = X / 255.0
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
                random_st = random.randint(0, 51 - BATCH_SIZE)
                val_acc, val_loss = test(test_X[random_st:random_st+BATCH_SIZE], test_y[random_st:random_st+BATCH_SIZE], net, device)
                f.write(
                    f"{MODEL_NAME},{round(time.time(), 3)},{round(float(acc), 2)},{round(float(loss), 4)},"
                    f"{round(float(val_acc), 2)},{round(float(val_loss), 4)},{epoch}\n")

        print(f"Epoch: {epoch}. Loss: {loss}")

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
