import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8 * 8 * 32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.dropout = nn.Dropout(p=0.40)
        self.fc1 = nn.Sequential(
            nn.Linear(8 * 8 * 64, 128),
            nn.ReLU())
        self.fc2 = nn.Linear(128, 10)

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.dropout(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out