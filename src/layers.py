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
import random
from xgboost import XGBClassifier
import math
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import f1_score,accuracy_score

hidden = 64
drop = 0.5
class Feature(nn.Module):

    def __init__(self,):
        super(Feature, self).__init__()
        self.l1 = nn.Conv1d(4, hidden, 5, stride = 2)
        self.pool1 = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.gru1 = nn.LSTM(hidden, hidden, 1, bidirectional=True)

        self.l2 = nn.Conv1d(hidden*2 + hidden, hidden, 3, stride = 1)
        self.pool2 = nn.MaxPool1d(2)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.gru2 = nn.LSTM(hidden, hidden, 1, bidirectional=True)

        self.l3 = nn.Conv1d(hidden*2 + hidden, hidden, 3, stride = 1)
        self.pool3 = nn.MaxPool1d(2)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.gru3 = nn.LSTM(hidden, hidden, 1, bidirectional=True)

        self.l4 = nn.Conv1d(hidden*2 + hidden, hidden, 1)

        self.adpool = nn.AdaptiveMaxPool1d(1)

        self.drop = nn.Dropout(drop)
    def forward(self, seq):

        #seq.shape = S * L * c -> S * c * l
        seq = seq.permute(0, 2, 1)

        s1 = self.l1(seq)
        s1 = self.bn1(s1)
        s1= self.pool1(F.relu(s1))
        # n c l -> l n c
        s1_t = s1.permute(2, 0, 1)
        s1_t,_ = self.gru1(s1_t)
        # l n c -> n c l
        s1_t = s1_t.permute(1, 2, 0)
        #print(s1_t.size(), s1.size())
        s1=torch.cat([s1_t, s1], axis=1)

        seq = s1

        s1 = self.drop(self.l4(s1))

        # S * C * L -> S * C
        #print(s1.size())
        s1 = self.adpool(s1)
        s1 = s1.squeeze(-1)
        return s1


class DeepRCI_conv(nn.Module):

    def __init__(self):

        super(DeepRCI_conv, self).__init__()
        self.feature = Feature()

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2  =nn.Linear(hidden, hidden)
        self.fc3  =nn.Linear(hidden, hidden)

        self.hidden = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(drop)

        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.out = nn.Linear(hidden, 1)

    def forward(self, seq):
        #特征提取阶段
        feature = self.feature(seq)

        hidden  = self.drop(F.relu(self.hidden(feature)))
        #with shortcut
        hidden = hidden + feature
        out = torch.sigmoid(self.out(hidden))
        return out


class Variformer(nn.Module):

    def __init__(self):

        super(Variformer, self).__init__()
        self.feature = Feature()

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2  =nn.Linear(hidden, hidden)
        self.fc3  =nn.Linear(hidden, hidden)

        self.hidden = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(drop)

        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.out = nn.Linear(hidden, 1)

    def forward(self, seq):
        # 1 * S * L * C  -> S * L * C
        seq = seq.squeeze(0) #
        #特征提取阶段
        feature = self.feature(seq)
        #print(feature[0][0])
        f1 = F.relu(self.fc1(feature))
        f2 = F.relu(self.fc2(feature))
        f3 = F.relu(self.fc3(feature))

        # variable self attention

        # basic assumpltion: will positional encoding matters????
        # good analysis
        alpha = f1.matmul(f2.permute(1,0)) # S * C x C * S
        alpha = F.softmax(alpha, dim = 1)
        attend_feature = alpha.matmul(f3)# S * S x S * C -> S * C

        # 用sum 还是用mean?
        # 不可能用sum, 长度会受到影响的！！！
        final = torch.mean(attend_feature, axis=0)

        # classifier
        hidden  = self.drop(F.relu(self.hidden(final)))
        #with shortcut
        hidden = hidden + final
        out = torch.sigmoid(self.out(hidden))
        return out

class DeepCRI(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Variformer()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        pred = self.model(x)
        return pred

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        pred = self.model(x)
        loss = F.l1_loss(pred, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = F.l1_loss(pred, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


# ## 训练代码

# # 先读取数据
# datadir='/data/group/data/rna_chromotin_interaction/iMAGIC/src/baseline_data/'
# fold = 0
# seqs = []
# labels = []
# for record in SeqIO.parse(f'{datadir}/train_{fold}.fasta','fasta'):
#     seqs.append(record.seq)
#     labels.append(int(record.id.split('_')[-1]))
#
# x_test = []
# y_test = []
# for record in SeqIO.parse(f'{datadir}/test_{fold}.fasta','fasta'):
#     x_test.append(record.seq)
#     y_test.append(int(record.id.split('_')[-1]))
# x_train, x_valid, y_train, y_valid = train_test_split(seqs, labels,random_state = seed)
# trainDataset = SeqDataset(x_train, y_train)
# validDataset = SeqDataset(x_valid, y_valid)
# testDataset = SeqDataset(x_test, y_test)

# In[30]:


from sklearn.metrics import accuracy_score,f1_score, precision_score, roc_auc_score, recall_score,average_precision_score
# check acc
thresh = 0.5
def get_metrics(preds, labels):
    #print(preds)
    #print(labels)
    acc = accuracy_score(labels, preds>thresh)
    sn = precision_score(labels, preds>thresh)
    sp = recall_score(labels, preds>thresh)
    auroc = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)
    f1 = f1_score(labels, preds>thresh)
    return [acc, sn, sp, auroc, auprc, f1]


# In[40]:

