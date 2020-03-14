import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
# from SpykeTorch import snn
# from SpykeTorch import functional as sf
# from SpykeTorch import visualization as vis
# from SpykeTorch import utils
# from torchvision import transforms
# import struct
# import glob


#https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627

numberOfImagesToDownload = 500; #multiply by 10, so 500 ~= 50 images

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=numberOfImagesToDownload, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=numberOfImagesToDownload, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

#print(images.shape)
#print(labels.shape)

#plt.imshow(images[0].numpy().squeeze(), cmap='gray_r');
#plt.show()

imagesWithSix = np.empty((numberOfImagesToDownload,28,28))
count = 0


#find the images with label '6' and store in array
for i in range(0, images.shape[0]):
    if labels[i] == 6:
        imagesWithSix[count,:,:] = images[i,0,:,:]
        count += 1

print(count, " images in numpy array with shape ", imagesWithSix.shape)


#turn grayscale to black/white mask
for i in range(0,count):
    for j in range(0,28):
        for k in range (0,28):
            if (imagesWithSix[i,j,k] != -1):
                imagesWithSix[i,j,k] = 1
mask = imagesWithSix


#print a few of the images
figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(mask[index], cmap='gray_r')
plt.show()



