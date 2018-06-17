import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from datagen import datagen
import util

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable

import os

np.set_printoptions(linewidth=200)

BATCH_SIZE  = 64
DATA_SIZE   = 2000
TEST_SIZE   = 10000
MAX_ERRORS  = 4

NUM_EPOCH = 50

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.00001

PRINT_EVERY = 100


print("Train data")
trainset = datagen(DATA_SIZE, seed=2018, maxErr=MAX_ERRORS)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print("Test data")
testset = datagen(TEST_SIZE, seed=1231231, maxErr=MAX_ERRORS)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(1, 4, (3,8), bias = False, stride = 1)
        #self.fc1 = nn.Linear(25*4, 1)
        self.conv1 = util.BinConv2D(1,4,(3,8))
        self.fc1 = util.BinLinear(25*4, 1)

    def forward(self, x):
        #print("\n\nshape of x")
        #print(np.shape(x))
        x, error1 = self.conv1(x)
        #print(np.shape(x))
        #x = F.relu(x)
        #print(np.shape(x))
        x = x.view(-1, 25 * 4)
        #print(np.shape(x))
        x, error2 = self.fc1(x)
        #print(np.shape(x))
        return x, error1, error2



net = Net()



criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

ONEP = 0
ONEN = 0


for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times

    running_loss = 0.0
    running_mse = 0.0
    running_distance = 0.0
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
        outputs, error1, error2 = net(inputs)
        
        # TODO komentirati ovu tehniku

        if epoch < 20:
            distanceFrom1Cost = WEIGHT_DECAY * util.DistanceFromPenalty(net.parameters(),1)
            MSEcost = criterion(outputs, labels)
            loss = MSEcost
        else:
            distanceFrom1Cost = WEIGHT_DECAY * util.DistanceFromPenalty(net.parameters(),1)
            MSEcost = criterion(outputs, labels)
            loss = MSEcost + distanceFrom1Cost

        loss.backward()
        optimizer.step()

        
        # #print statistics
        running_loss        += loss.item()
        running_mse         += MSEcost.item()
        running_distance    += distanceFrom1Cost.item()
        if i % PRINT_EVERY == PRINT_EVERY-1:    # print every 2000 mini-batches
            print('[%2d, %5d] loss: %.4f  MSE: %.4f  Dist: %.4f' %
                  (epoch + 1, i + 1, running_loss / PRINT_EVERY, running_mse / PRINT_EVERY, running_distance / PRINT_EVERY))
            running_loss     = 0.0
            running_mse      = 0.0
            running_distance = 0.0

print("ONEP = %d\nONEN = %d\n" % (ONEP,ONEN))

print('Finished Training')

ONEP = 0
ONEN = 0

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data

        inputs = data["inputs"]
        labels = data["labels"]

        outputs, error1, error2 = net(inputs)
        #print(labels-outputs)
        absErr = np.abs(labels-outputs)
        select = np.where(absErr < 0.5, 1, 0)
        

        #print("input %f - %f output" % (labels, outputs))
        total += labels.size(0)
        correct += np.count_nonzero(select)
# TODO see as to on which parameters it works bad when eight decay is introduced
print('Accuracy of the network on the 10000 test images: %f %%' % (
    100 * correct / total))

np.set_printoptions(linewidth=200)
print("\nParam data")
for param in net.parameters():
  print(param.data)


for param in net.parameters():
  param.data = torch.sign(param)

np.set_printoptions(linewidth=200)
print("\nParam data")
for param in net.parameters():
  #print(param.data)
  pass
