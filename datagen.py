
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from warnings import warn

# For the repeatability
random.seed(2018)

def seq2bit(seq):
    seq = np.array(seq, dtype = np.uint8)
    bitseq = []
    flag = 0
    for nuber in seq:
        bits = np.unpackbits(nuber)
        bits = bits[-3:]
        bits = np.vstack(bits)
        if flag == 0:
            bitseq = bits
            flag = 1
        else:
            bitseq = np.concatenate((bitseq, bits),1)
    return bitseq




def addErr(bitseq, numErr = 1):
    """Adds an error (flips a bit) in an input sequence
    
    Arguments:
        bitseq {array} -- Array of 0 and 1 in which you want to add an error
    
    Keyword Arguments:
        numErr {int} -- Number of errors you want to introduce (default: {1})
    """
    seq = np.copy(bitseq)
    a = np.shape(seq)

    for i in range(0,numErr):

        x = random.randint(0,a[0]-1)
        y = random.randint(0,a[1]-1)

        if seq[x][y] == 0:
            seq[x][y] = 1
        elif seq[x][y] == 1:
            seq[x][y] = 0
        else:
            warn("This shouldunt be anything othera than 0 or 1, it is %d" % seq[x][y])

    return seq



def paddingSeq(width):
    """ Generates sequence of bits size of width x 3
    
    Arguments:
        width {int} -- [description]
    """
    seq = []
    for i in range(width):
        num = np.array(random.getrandbits(3), dtype=np.uint8)
        bits = np.unpackbits(num)
        bits = bits[-3:]
        bits = np.vstack(bits)
        if i == 0:
            seq = bits
        else:
            seq = np.concatenate((seq, bits),1)
    return seq


def returnAllRotations(seq, totalLength = 32):
    inLength = np.shape(seq)[1]
    seqLength = totalLength - inLength
    print("\nReturn all rotations")
    print(seqLength)
    if seqLength <= 0:
        warn("Total sequence length <= 0")
    padding = paddingSeq(seqLength)
    padding = np.zeros((3,seqLength))
    padded = np.concatenate((seq, padding), axis=1)
    bulk = []

    for i in range(0,seqLength + 1):
        bulk.append(np.roll(padded,i,axis=1))
    return np.stack(bulk, axis = 0)


class datagen(Dataset):
    def __init__(self, numSamples=10000, seed=2018, maxErr = 1, errP = 0.5):
        """
        Args:
            numSamples: number of samples to generate
        """
        self.sequence = [5, 7, 0, 3, 6, 6, 4, 7, 5, 0, 4, 2]
        self.bitseq = seq2bit(self.sequence)
        
        self.numSamples = numSamples
        self.colectedSamples = numSamples
        self.seed = seed
        self.maxErr = maxErr
        self.errP = errP

        self.generateData()

        

    def generateData(self):
        random.seed(self.seed)
        negativeSamples  = self.numSamples//2

        self.data = []
        self.output = []

        for i in range(negativeSamples):
            self.colectedSamples -= 1
            self.data.append(paddingSeq(32))
            self.output.append(-1)

        #self.data = np.stack(self.data, axis = 0)
        #print(np.shape(self.data))


        while self.colectedSamples > self.numSamples//4:
            self.returnAllRotations(self.bitseq)

        while self.colectedSamples > 0:
            errBiteseq = addErr(self.bitseq, 1)
            self.returnAllRotations(errBiteseq)

        print("shape of data")
        print(np.shape(self.data))
        print(np.shape(self.output))
        print(self.colectedSamples)




    def returnAllRotations(self, seq, totalLength = 32):
        inLength = np.shape(seq)[1]
        seqLength = totalLength - inLength
        if seqLength <= 0:
            warn("Total sequence length <= 0")
        padding = paddingSeq(seqLength)
        padding = np.zeros((3,seqLength))
        padded = np.concatenate((seq, padding), axis=1)
        bulk = []

        for i in range(0,seqLength + 1):
            self.data.append(np.roll(padded,i,axis=1))
            
            self.output.append(1)
            self.colectedSamples -= 1

        #stack = np.stack(bulk, axis = 0)
        #self.data = np.concatenate((self.data, stack))



        


    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
    
        sample = {'data': self.data[idx], 'output': self.output[idx]}


        return sample




x = datagen()
print(len(x))
