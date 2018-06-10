
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# For the repeatability
random.seed(2018)

news = []
for i in range(1,20):
    numbers = np.array(random.getrandbits(3), dtype=np.uint8)
    bits = np.unpackbits(numbers)
    bits = bits[-3:]
    bits = np.vstack(bits)
    print(np.shape(bits))
    if i == 1:
        news = bits
    else:
        news = np.concatenate((news, bits),1)
    print(np.transpose(bits))

print(news)

"""
seq2bit: returns bit vector of input array
"""
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


class datagen(Dataset):
    def __init__(self, numSamples=10000, seed=2018, maxErr = 1, errP = 0.5):
        """
        Args:
            numSamples: number of samples to generate
        """

        self.sequence = [5, 7, 0, 3, 6, 6, 4, 7, 5, 0, 4, 2]
        self.bit2seq = seq2bit(self.sequence)
        
        self.numSamples = numSamples
        self.seed = seed
        self.maxErr = maxErr
        self.errP = errP


        for i in range(0,numSamples-1):
            pass





        


    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


