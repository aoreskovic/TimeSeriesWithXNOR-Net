import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from datagen import datagen

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable



BATCH_SIZE  = 64
DATA_SIZE   = 100000
TEST_SIZE   = 10000
MAX_ERRORS  = 100

NUM_EPOCH = 4

PRINT_EVERY = 100

trainset = datagen(DATA_SIZE, seed=2018, maxErr=MAX_ERRORS)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testset = datagen(TEST_SIZE, seed=2019, maxErr=MAX_ERRORS)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (3,8), bias = False, stride = 1)
        self.fc1 = nn.Linear(25*4, 1)

    def forward(self, x):
        #print("\n\nshape of x")
        #print(np.shape(x))
        x = self.conv1(x)
        #print(np.shape(x))
        x = F.relu(x)
        #print(np.shape(x))
        x = x.view(-1, 25 * 4)
        #print(np.shape(x))
        x = self.fc1(x)
        #print(np.shape(x))
        return x


net = Net()



criterion = nn.MSELoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

ONEP = 0
ONEN = 0


for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        inputs = data["inputs"]
        labels = data["labels"]
        
        #print("inputs/outputs")
        #print(np.shape(inputs))
        #print(np.shape(labels))

        

        for number in labels:
            
            if number == 1:
                ONEP += 1
            elif number == -1:
                ONEN += 1
            else:
                Warning("dasdasdasdasdasa")



        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        
        # TODO ispect this
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        
        # #print statistics
        running_loss += loss.item()
        if i % PRINT_EVERY == PRINT_EVERY-1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / PRINT_EVERY))
            running_loss = 0.0

print("ONEP = %d\nONEN = %d\n" % (ONEP,ONEN))

print('Finished Training')



correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data

        inputs = data["inputs"]
        labels = data["labels"]

        outputs = net(inputs)
        #print(labels-outputs)
        absErr = np.abs(labels-outputs)
        select = np.where(absErr < 0.5, 1, 0)

        

        #print("input %f - %f output" % (labels, outputs))
        total += labels.size(0)
        correct += np.count_nonzero(select)

print('Accuracy of the network on the 10000 test images: %f %%' % (
    100 * correct / total))

