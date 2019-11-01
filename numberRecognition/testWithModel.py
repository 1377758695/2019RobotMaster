import torch
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from myDataset import CNN
import matplotlib.pyplot as plt


batch_size = 64
epochs = 20
log_interval = 10


class myDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform = None):
        f = open(txt_path, 'r')
        imgs = []
        for line in f:
            line = line.rstrip()
            words = line.split()
            print(words)
            imgs.append((words[0], int(words[1])))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, item):
        f, label = self.imgs[item]
        img = cv2.imread(f)
        img = cv2.resize(img, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[:, :, None]
        if self.transform is not None:
            img = self.transform(img)
        img.unsqueeze(0)
        return img, label

    def __len__(self):
        return len(self.imgs)

# eval_losses = []
eval_acces = []
eval_misses = []
eval_wrongs = []

def test(epoch):
    # eval_loss = 0
    eval_wrong = 0
    eval_miss = 0
    eval_acc = 0
    model.eval()
    for batch, (img, label) in enumerate(test_loader):
        img = Variable(img)
        label = Variable(label)

        output = model(img)

        # prob = F.softmax(output, dim=1)
        # prob = Variable(prob)
        # prob = prob.numpy()
        # print(prob)
        # max_prob = np.max(prob)
        # pred = np.argmax(prob)

        # eval_loss += loss.item()
        miss = 0
        _, pred = output.max(1)
        # print('pred: {}'.format(pred))
        # print('label: {}'.format(label))
        num_correct = (pred == label).sum().item()
        wrong = (pred != label).sum().item()
        acc = num_correct / len(pred)

        for i in range(len(pred)):
            if pred[i] == 10 and label[i] < 10:
                miss += 1

        eval_acc += acc
        eval_miss += miss
        eval_wrong += wrong

        if batch % log_interval == 0 and wrong != 0:
            print('Test Epoch : {} [{}/{} ({:.0f}%)]\tAccuracy: {:.6f}\tWrong: {}\tMiss: {} ({:.2f}%)'.format(
                epoch+1, batch*len(img), len(test_loader.dataset),
                100.*batch/len(test_loader), acc, wrong, miss, 100.*miss/wrong
            ))
        elif batch % log_interval == 0 and wrong == 0:
            print('Test Epoch : {} [{}/{} ({:.0f}%)]\tAccuracy: {:.6f}\tWrong: {}\tMiss: {}'.format(
                epoch+1, batch*len(img), len(test_loader.dataset),
                100.*batch/len(test_loader), acc, wrong, miss
            ))
    # eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    eval_wrongs.append(eval_wrong / len(test_loader))
    eval_misses.append(eval_miss/len(test_loader))

    print('Epoch {} Test Accuracy {} Test Wrong {} Test Miss {}'.format(
        epoch+1, eval_acc/len(test_loader), eval_wrong/len(test_loader), eval_miss/len(test_loader)
    ))


if __name__ == '__main__':
    model = torch.load('model2.pth')
    model.eval()

    '''
    img = cv2.imread("D:/theThirdYear/RM/task1/gray/0502.jpg")
    img = cv2.resize(img, (28, 28))
    trans = transforms.Compose(
        [
            # transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[:, :, None]
    img = trans(img)
    img = img.unsqueeze(0)
    '''

    test_dataset = myDataset(txt_path="D:/theThirdYear/RM/task1/label.txt",
                             transform=transforms.Compose(
                                 [
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]
                             ))

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=True)
    for epoch in range(epochs):
        test(epoch)
    # print(img.size())
    '''
    output = model(img)
    prob = F.softmax(output, dim=1)
    prob = Variable(prob)
    prob = prob.numpy()
    print(prob)
    max_prob = np.max(prob)
    pred = np.argmax(prob)
    print('{}, {}'.format(max_prob, pred.item()))
    if (max_prob > 0.8):
        print(pred.item())
    else:
        print(str(-1))
    '''


# img = plt.open("D:/theThirdYear/RM/task1/gray/1600.jpg")
