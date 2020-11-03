import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# Train data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST('data', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)

# Test data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_set = datasets.MNIST('data', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)

# Visualize example
images, labels = next(iter(train_loader))
print(images.shape)
img = images[0].numpy().squeeze()
print(img.shape)
plt.figure(), plt.imshow(img, cmap='gray')


# CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)  # 28x28x1 * (5x5x1)x16 = 28x28x16
        self.max_pool1 = nn.MaxPool2d(2, 2)  # 28x28x16 * (2,2) = 14x14x16
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)  # 14x14x16 * (5,5,16)x32 = 14x14x32
        self.max_pool2 = nn.MaxPool2d(2, 2)  # 14x14x32 * (2,2) = 7x7x32
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # first cnn
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))  # second cnn
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)  # flatten output
        x = self.fc(x)  # fully connected
        return x

# Training CNN

