import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import util
from datagen import datagen

from matplotlib import pyplot as plt

np.set_printoptions(linewidth=300)


# ---------- Hyperparameters ----------

BATCH_SIZE = 64
DATA_SIZE = 20000
TEST_SIZE = 10000
MAX_ERRORS = 6

NUM_EPOCH = 200
REGULARIZATION_START = 40

LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.002

PRINT_EVERY = 50


# ---------- Loading data ----------

print("Train data")
trainset = datagen(DATA_SIZE, seed=2018, maxErr=MAX_ERRORS)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print("Test data")
testset = datagen(TEST_SIZE, seed=1231231, maxErr=MAX_ERRORS)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ---------- Defining the network ----------

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = util.BinConv2D(1, 4, (3, 8), bias=True)
        self.fc1 = util.BinLinear(25*4, 1, bias=True)


    def forward(self, x):
        x, error1 = self.conv1(x)
        x = x.view(-1, 25 * 4)
        x, error2 = self.fc1(x)
        return x, error1, error2


net = Net()


# ---------- Cost function ----------

# Using the Mean Squared Error loss function
criterion = nn.MSELoss()

# And using the ADAM oprimizer
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE,
                       betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


best_result = 100000

plt.axis('tight')
plt.ylim(ymin=0)
plt.ion()
plt.show()



array_loss = list()
array_MSE = list()
array_distance = list()



# ---------- Training ----------
for epoch in range(NUM_EPOCH):

    running_loss = 0.0
    running_mse = 0.0
    running_distance = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        inputs = data["inputs"]
        labels = data["labels"]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, error1, error2 = net(inputs)

        # TODO komentirati ovu tehniku

        if epoch < REGULARIZATION_START:
            distanceFrom1Cost = WEIGHT_DECAY * util.DistanceFromPenalty(net.parameters(), 1)
            MSEcost = criterion(outputs, labels)
            loss = MSEcost
        else:
            distanceFrom1Cost = WEIGHT_DECAY * util.DistanceFromPenalty(net.parameters(), 1)
            MSEcost = criterion(outputs, labels)
            loss = MSEcost + distanceFrom1Cost

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_mse += MSEcost.item()
        running_distance += distanceFrom1Cost.item()
        if i % PRINT_EVERY == PRINT_EVERY-1:   
            print('[%2d, %5d] loss: %.4f  MSE: %.4f  Dist: %.4f' %
                  (epoch + 1, i + 1, running_loss / PRINT_EVERY, running_mse / PRINT_EVERY, running_distance / PRINT_EVERY))

            
            array_loss.append(running_loss / PRINT_EVERY)
            array_MSE.append(running_mse / PRINT_EVERY)
            array_distance.append( running_distance / PRINT_EVERY)
            x_axis = range(len(array_loss))


            plt.clf()
            plt.axis([0, max(x_axis), 0, 1])
            plt.title("Prikaz funkcija cijena")
            lines = plt.plot(x_axis, array_MSE, 'b', x_axis, array_loss, 'r',  x_axis, array_distance, 'g')
            plt.setp(lines, linewidth=0.8, antialiased=True)
            plt.legend(('MSE', 'Ukupna cijena', 'Udaljenost od +-1'))
            plt.draw()
            plt.pause(0.01)


            running_loss = 0.0
            running_mse = 0.0
            running_distance = 0.0

            
            #util.PrintDistanceFrom(net.parameters(),1)

        

        if running_loss < best_result:
            is_best = True
            best_result = running_loss

"""
        util.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_prec1': best_result,
            'optimizer': optimizer.state_dict(),
        }, is_best)
"""



print('Finished Training')


# ---------- Testing on validation set ----------

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data

        inputs = data["inputs"]
        labels = data["labels"]

        outputs, error1, error2 = net(inputs)

        absErr = np.abs(labels-outputs)
        select = np.where(absErr < 0.5, 1, 0)

        total += labels.size(0)
        correct += np.count_nonzero(select)


# ---------- Printing the accuracy and results ----------

print('Accuracy of the network on the 10000 test images: %f %%' % (
    100 * correct / total))


print("\nParam data")
for param in net.parameters():
    print(param.data)

i = 0
for param in net.parameters():
    if (i % 2) == 0:
        param.data = torch.sign(param)
    else:
        #param.data = torch.round(param)ž
        pass
    i +=1


print("\nParam data")
for param in net.parameters():
    print(param.data)


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data

        inputs = data["inputs"]
        labels = data["labels"]

        outputs, error1, error2 = net(inputs)

        absErr = np.abs(labels-outputs)
        select = np.where(absErr < 0.5, 1, 0)

        total += labels.size(0)
        correct += np.count_nonzero(select)


# ---------- Printing the accuracy and results ----------

print('Accuracy of the binarized network on the 10000 test images: %f %%' % (
    100 * correct / total))

input("Press [enter] to continue.")
