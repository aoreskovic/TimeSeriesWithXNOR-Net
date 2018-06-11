import random
from warnings import warn

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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
    """Adds an error (flips a bit) in an input sequence
    
    Arguments:
        bitseq {array} -- Array of 0 and 1 in which you want to add an error
    
    Keyword Arguments:
        numErr {int} -- Number of errors you want to introduce (default: {1})
    """
    seq = np.copy(bitseq)
    a = np.shape(seq)
    points = []
    i = 0

    if numErr > np.size(bitseq) * 0.6:
        warn("Lot of errors, may get stuck in while")

    while i < numErr:
        x = random.randint(0,a[0]-1)
        y = random.randint(0,a[1]-1)
        
        if (x,y) in points:
            continue
        else:
            points.append((x,y))
        i += 1

    for (x,y) in points:

        if seq[x][y] == 0:
            seq[x][y] = 1
        elif seq[x][y] == 1:
            seq[x][y] = 0
        else:
            warn("This shouldunt be anything othera than 0 or 1, it is %d" % seq[x][y])

    return seq


print("\n\n seq2bit:\n")
bitseq = seq2bit(sequence)
print(bitseq)
print()
arry = np.array(addErr(bitseq,20),dtype=np.int8)
#print(arry)
print("shift right")
print(np.roll(arry,1,axis=1))
print("bitesq after roll and god knows")
print(bitseq)


def paddingSeq(width):
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

#print("\nPAdding seq 3") 
#print(paddingSeq(6))

#print("padding")

padded = np.concatenate((arry, paddingSeq(20)), axis=1)
#print(padded)
#print(np.shape(padded))

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


print("bulking")
bulk = []
print("bitseq")
print(bitseq)
for i in range(0,20):
    padded = np.concatenate((bitseq, np.zeros((3,20))), axis=1)
    bulk.append(np.roll(padded,i,axis=1))


np.set_printoptions(linewidth=200)
print("bulk shape")
print(np.shape(bulk))
#print(np.shape(np.stack(bulk, axis = 0)))
#print(np.stack(bulk, axis = 0))


print("\n return all rotations")
bulfrm = returnAllRotations(bitseq)
print(np.shape(bulfrm))
print(bulfrm[0])
print(bulfrm[-1])


paddd = paddingSeq(32)
print(paddd)
print(np.shape(paddd))


print("bitseq")
print(bitseq)
errbitseq =addErr(bitseq,30)
print(errbitseq)
print(bitseq-errbitseq)


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


