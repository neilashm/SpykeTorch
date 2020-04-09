import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def makeGray(length, precision=10):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=length, shuffle=False)
    

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    gray = 0.2989 * images[:,0,:,:] + 0.5870 * images[:,1,:,:] + 0.1140 * images[:,2,:,:] #this is matlab's grayscale algo, may need a different one
    #uncomment 3 lines below to see how bad that gray scale formula look
    #print(gray.size())
    #plt.imshow(gray[0].numpy().squeeze(), cmap='gray_r');
    #plt.show()

    gray.resize_((gray.shape[0], gray.shape[1]*gray.shape[2])) #reshape to 2d
    #print(gray.size())

    temporal = torch.zeros(gray.shape[0], gray.shape[1], precision)
    #print(temporal.size()) torch.Size([5, 1024, 10])
    for i in range(0,gray.shape[0]):
        values = gray[i,:].numpy() +1 #between 0..2
        values *= (precision/2)
        values = values.astype(int)

        for j in range(0, gray.shape[1]):
            temporal[i,j,values[j]] = 1

    return temporal


print(makeGray(length=5, precision=10).size())
# output dimensions (length, 1024, precision)
# length = number of images in set
# 1024 = 32*32 pixels per image
# precision  = how long your temporal 0,1 train is
#exit()