# virtual env:  graph
# %%
import torch
import os
import pickle as pkl
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable
import numpy as np
from utils import *
from const import *
import random

# %%
np.random.seed(65535)
random.seed(65535)

print("load dna data from file")
X_train, y_train = readFileNumpy(datadir + train_name)

X_valid, y_valid = readFileNumpy(datadir + valid_name)


class SeqDataset(Dataset):
    def __init__(self, files):
        self.files = files
        self._parse_data()
        self.len = self.x_data.shape[0]

    def _parse_data(self, ):
        self.x_data = self.files[0]
        self.y_data = self.files[1]

    def __getitem__(self, index):
        rna = self.x_data[index][:101]
        dna = self.x_data[index][101:]
        label = self.y_data[index]
        rna_pe = positionalEncoding(101)
        dna_pe = rna_pe
        return rna, rna_pe, dna, dna_pe, label

    def __len__(self):
        return self.len


trainDataset = SeqDataset((X_train, y_train))
validDataset = SeqDataset((X_valid, y_valid))

if __name__ == '__main__':
    train_loader = DataLoader(dataset=trainDataset,
                              batch_size=32,
                              shuffle=True)

    valid_loader = DataLoader(dataset=validDataset,
                              batch_size=32,
                              shuffle=True)
