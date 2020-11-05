import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# Train data
batch_size = 128
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST('data', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Test data
test_set = datasets.MNIST('data', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Visualize example
images, labels = next(iter(train_loader))
print(images.shape)
img = images[0].numpy().squeeze()
print(img.shape)
plt.figure(), plt.imshow(img, cmap='gray')


# CNN
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Training CNN
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('CUDA is available! Training on GPU...')
else:
    print('CUDA is not available. Training on CPU...')

model = Classifier()
if train_on_gpu:
    model.cuda()

epochs = 5
learning_rate = 0.1
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def training(_epochs, _model, _train_loader, _test_loader, _test_set, _train_on_gpu=True):
    _train_loss_l, _test_loss_l = [], []
    accuracy = None
    for epoch in range(_epochs):
        train_loss = 0
        test_loss = 0
        # Train
        for _images, _labels in _train_loader:
            if _train_on_gpu:
                _images, _labels = _images.cuda(), _labels.cuda()
            optimizer.zero_grad()
            output = model(_images)
            loss = criterion(output, _labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        _train_loss_l.append(train_loss / len(_train_loader))

        # Test
        correct = 0
        model.eval()
        for _images, _labels in _test_loader:
            if _train_on_gpu:
                _images, _labels = _images.cuda(), _labels.cuda()
            output = model(_images)
            loss = criterion(output, _labels)
            test_loss += loss.item()

            _, yhat = torch.max(output, 1)
            correct += (yhat == _labels).sum().item()

        accuracy = correct / len(_test_set)
        _test_loss_l.append(test_loss / len(_test_loader))
        # Prints
        print('Epoch: {}/{} \tTrain loss:{:.3} \tTest loss:{:.3}'.format(epoch + 1, _epochs,
                                                                         train_loss / len(_train_loader),
                                                                         test_loss / len(_test_loader)))
    print('\nGlobal accuracy: {:.1%}'.format(accuracy))
    return _train_loss_l, _test_loss_l


# train_loss, test_loss = training(epochs, model, train_loader, test_loader, test_set)
# plt.figure()
# plt.plot(train_loss, label='Training loss')
# plt.plot(test_loss, label='Test loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# torch.save(model.state_dict(), 'classifier_mnist.pt')

state_dict = torch.load('classifier_mnist.pt')
model.load_state_dict(state_dict)

# import cv2
# img = cv2.imread('num.jpg', 0)
# img = cv2.resize(img, (28, 28))
# img = img.reshape(1, 1, 28, 28)
# img = torch.from_numpy(img)
# img = img.type('torch.FloatTensor')
# output = model(img.cuda())
# _, pred = torch.max(output, 1)

img_test = images[0]
plt.figure(), plt.imshow(img_test.numpy().squeeze())
img_test = torch.unsqueeze(img_test, 0).type('torch.FloatTensor')
if train_on_gpu:
    img_test = img_test.cuda()
output = model(img_test)
_, pred = torch.max(output, 1)
print(pred)
