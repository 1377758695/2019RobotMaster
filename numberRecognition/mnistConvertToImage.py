import os
from skimage import io
import torchvision.datasets.mnist as mnist
import cv2
import numpy

root = "D:/DeepLearning/Dataset/mnist_dataset/raw"

train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
)

test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
)

print("train set:", train_set[0].size())
print("test set:", test_set[0].size())


def convert_to_img(train=True):
    if(train):
        f = open(root + '/train/label.txt', 'w')
        data_path = root + '/train/image/'
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
            img_path = data_path + str(label.item()) + '_' + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label.item()) + '\n')
            # print(str(label.item()) + '_' + str(i) + '.jpg' + '\n')
            # print(str(label) + '\n')
            cv2.waitKey()
        f.close()
    else:
        f = open(root + '/test/label.txt', 'w')
        data_path = root + '/test/image/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            img_path = data_path + str(label.item()) + '_' + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label.item()) + '\n')
            # print(str(label.item()) + '_' + str(i) + '.jpg')
            # print(img_path + ' ' + str(label) + '\n')
            cv2.waitKey()
        f.close()


convert_to_img(True)
convert_to_img(False)