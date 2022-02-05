#!/usr/bin/env python
# coding: utf-8

from numpy.lib.financial import nper
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
import os
from Bio import SeqIO
import torch
import os
import torch
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
#import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import  ModelCheckpoint
from triplex_util import *
import os
import torch.nn.functional as F
from torch import nn
import pickle as pkl
import numpy as np
from utils import *
import os
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.metrics import f1_score,accuracy_score
from const import *
from layers import *
import sys

baseLen = int(sys.argv[1])
experiment = sys.argv[2]

fold= int(sys.argv[3])
c  = int(sys.argv[4])

datadir = sys.argv[5]

device= str(c % 4)
os.environ['CUDA_VISIBLE_DEVICES']=device

exp = f'{experiment}_{baseLen}_{hidden}_{seed}_{lr}'
os.system(f'zip -r code_bkp/{exp}.zip *.py')

# # DeepCRI 模型设计
seed_all(c)
def sequenceToTensor(seq):
    #
    rs = len(seq)// baseLen + 1
    tensor = []
    for i in range(rs):
        s = seq[i * baseLen: (i+1)* baseLen]
        s = s + "N" * max(baseLen - len(s), 0)
        tensor.append(stringOnehot(s))
    return np.array(tensor)

# use generator
class SeqDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = sequenceToTensor(self.x[index])
        y = self.y[index]
        return np.array(x).astype(np.float32), y

    def __len__(self):
        return len(self.x)

def evaluate(model,valid_loader):
    model.eval()
    preds = []
    ys = []
    for x,y in valid_loader:
        x,y = x.cuda(), y.cuda()

        pred = model(x)
        ys.append(y.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
    preds = np.array(preds).reshape(-1)
    ys = np.array(ys).reshape(-1)
    acc = accuracy_score(preds>0.5, ys)
    result = get_metrics(preds, ys)
    return acc,result


def read(t,fold):
    x_test = []
    y_test = []
    for record in SeqIO.parse(f'baseline_data/{t}_{fold}.fasta','fasta'):
        x_test.append(str(record.seq))
        y_test.append(int(str(record.id).split('_')[-1]))
    return x_test, y_test


x_train,y_train = read(f'ensemble_fin_1000_64_0_train',fold)

x_valid, y_valid = read(f'ensemble_fin_1000_64_0_valid',fold)

x_test, y_test = read(f'ensemble_fin_1000_64_0_test',fold)

trainDataset = SeqDataset(x_train, y_train)

validDataset = SeqDataset(x_valid, y_valid)
testDataset = SeqDataset(x_test, y_test)

best_acc = -1
patience = 0
model = Variformer()
adam = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

model.to('cuda')

# 编写dataloader
train_loader = DataLoader(dataset=trainDataset,
                            batch_size=batch_size,
                            shuffle=False)
valid_loader = DataLoader(dataset=validDataset,
                            batch_size=batch_size,
                            shuffle=False)
test_loader = DataLoader(dataset=testDataset,
                            batch_size=batch_size,
                            shuffle=False)
for e in range(max_epoch):
    model.train()
    losses = []
    ys = []
    preds = []
    for x, y in train_loader:
        adam.zero_grad()
        x,y = x.cuda(), y.cuda()
        pred = model(x)

        loss = F.l1_loss(pred, y)

        loss.backward()
        adam.step()
        losses.append(loss.item())
        ys.append(y.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
    acc1 = accuracy_score(np.array(preds).reshape(-1,)>0.5, np.array(ys).reshape(-1))
    loss = np.mean(losses)

    acc2,_ = evaluate(model,valid_loader)
    #if e%25 == 0:
    print(f'epoch {e} train acc {acc1} loss {loss} valid {acc2}')
    if acc2 > best_acc:
        patience = 0
        best_acc = acc2
        torch.save(model, f'../checkpoint/best_{exp}_{fold}_{c}.hdf5')
    else:
        patience +=1
        if patience > 100:
            break
