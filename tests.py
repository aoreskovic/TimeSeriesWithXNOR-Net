import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from warnings import warn

# For the repeatability
random.seed(2018)

sequence = [5, 7, 0, 3, 6, 6, 4, 7, 5, 0, 4, 2]
print(sequence)

npseq = np.array(sequence, dtype=np.uint8)
npseq = np.hstack(npseq)
#print(npseq)

npseqbit = np.unpackbits(npseq, axis=0)
#print(npseqbit)



def seq2bit(seq):
    bitseq = []
    flag = 0
    for number in seq:
        number = np.array(number, dtype=np.uint8)
        bits = np.unpackbits(number)
        bits = bits[-3:]
        bits = np.vstack(bits)
        if flag == 0:
            bitseq = bits
            flag = 1
        else:
            bitseq = np.concatenate((bitseq, bits),1)
    return bitseq

def addErr(bitseq, numErr = 1):
    seq = bitseq
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
    
    



print("\n\n seq2bit:\n")
bitseq = seq2bit(sequence)
print(seq2bit(sequence))
print()
arry = np.array(addErr(bitseq,20),dtype=np.int8)
print(arry)





news = []
for i in range(1,20):
    numbers = np.array(random.getrandbits(3), dtype=np.uint8)
    bits = np.unpackbits(numbers)
    bits = bits[-3:]
    bits = np.vstack(bits)
    if i == 1:
        news = bits
    else:
        news = np.concatenate((news, bits),1)



def seq2bit(seq):
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