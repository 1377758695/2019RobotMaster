import torch
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from myDataset import CNN
import matplotlib.pyplot as plt


if __name__ == '__main__':
    model = torch.load('model3.pth')
    model.eval()


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

    # print(img.size())

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



# img = plt.open("D:/theThirdYear/RM/task1/gray/1600.jpg")
