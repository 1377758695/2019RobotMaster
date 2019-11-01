import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2

batch_size = 64
epochs = 6
lr = 0.001
log_interval = 10

# 实现单张图片可视化


def showBatch():
    images, labels = next(iter(train_loader))
    img = torchvision.utils.make_grid(images)

    img = img.numpy().transpose(1, 2, 0)
    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    img = img * std + mean
    print(labels)
    cv2.imshow('win', img)
    cv2.waitKey()

# 搭建神经网络


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(          #(28*28*1)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),                  #(28*28*16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)#(14*14*16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),  #(14*14*32()
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   #(7*7*32)
        )
        self.out = nn.Linear(32*7*7, 11)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


def train(epoch):
    train_loss = 0
    train_acc = 0
    model.train()
    for batch, (img, label) in enumerate(train_loader):
        img = Variable(img)
        label = Variable(label)

        output = model(img)
        loss = loss_func(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, pred = output.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]

        train_acc += acc

        if batch % log_interval == 0:
            print('Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch*len(img), len(train_loader.dataset),
                100.*batch/len(train_loader), loss.item()
            ))
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    print('Epoch {} Train Loss {} Train Accuracy {}'.format(
        epoch + 1, train_loss / len(train_loader), train_acc / len(train_loader)
    ))


def test(epoch):
    eval_loss = 0
    eval_acc = 0
    model.eval()
    for batch, (img, label) in enumerate(test_loader):
        img = Variable(img)
        label = Variable(label)

        output = model(img)
        loss = loss_func(output, label)

        eval_loss += loss.item()

        _, pred = output.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]

        eval_acc += acc

        if batch % log_interval == 0:
            print('Test Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch*len(img), len(test_loader.dataset),
                100.*batch/len(test_loader), loss.item()
            ))
    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))

    print('Epoch {} Test Loss {} Test Accuracy {}'.format(
        epoch+1, eval_loss/len(test_loader), eval_acc/len(test_loader)
    ))


if __name__ == '__main__':
    train_dataset = datasets.MNIST(root='D:/DeepLearning/Dataset/mnist_dataset',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)
    test_dataset = datasets.MNIST(root='D:/DeepLearning/Dataset/mnist_dataset',
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=True)

    model = CNN()
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    losses = []
    acces = []
    eval_losses = []
    eval_acces = []

    for epoch in range(epochs):
        train(epoch)
        test(epoch)

    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, 'model1.pth')

