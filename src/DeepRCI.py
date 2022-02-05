#import torch
import os
import pickle as pkl
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
from utils import *
from const import *
import os
import tensorflow as tf
import argparse
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import random
import pickle as pkl
from sklearn.metrics import accuracy_score,f1_score, precision_score, roc_auc_score, recall_score,average_precision_score
import os
from sklearn.metrics import f1_score,accuracy_score

def readFileNumpy(all_lines, is_debug=False):
    data_np = None
    data_size = debug_size if is_debug else len(all_lines)
    #data_np = np.zeros((data_size, input_length * 2, 4))
    labels = [
        0,
    ] * data_size
    rnas = []
    dnas = []
    features = []
    for index, line in enumerate(all_lines):
        if is_debug and index >= data_size:
            break
        lines = line.strip().split()
        rna = lines[0]
        dna = lines[1]
        label = lines[-1]
        f = [float(v) for v in lines[-4:-1]] # [-4:-1] [-3:-1]
        rna = rna[:input_length] + "N" * max(input_length - len(rna), 0)
        dna = dna[:input_length] + "N" * max(input_length - len(dna), 0)
        rnas.append(stringOnehot(rna))
        dnas.append(stringOnehot(dna))
        features.append(f)
        #data = stringOnehot(rna + dna)
        #data_np[index] = data
        labels[index] = float(label)
    return np.array(rnas), np.array(dnas), np.array(features), np.array(labels)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--experiment', type=str, default='bayesian',
                    help='experiment name')
parser.add_argument('--model', type=str,default='bayesian',
                    help='model name')
parser.add_argument('--hidden_dim', type=int,default=128,
                    help='hidden dimension')
parser.add_argument('--seed', type=int,default=65535,
                    help='random seed for reproduct')
parser.add_argument('--kernel_size', type=int,default=3,
                    help='convolutional kernel size')
parser.add_argument('--mode', type=str,default="test",
                    help='train or test')
parser.add_argument('--dataset', type=str, default='../data/stress20/', help='You dataset directory')

args = parser.parse_args()

#os.environ['CUDA_VISIBLE_DEVICES']= str(r%4)
exp = args.experiment
model_name = args.model
hidden_dim = args.hidden_dim
kernel_size = args.kernel_size
dataset = args.dataset
train_name = f'new_train.data'
valid_name = f'new_valid.data'
test_name = f'new_test.data'
seed = args.seed
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

input_len = 101
print("load dna data from file")
datadir=args.dataset
train_rna,train_dna, train_feature, y_train = readFileNumpy(open(datadir + train_name).readlines(),is_debug=False)
valid_rna, valid_dna, valid_feature, y_valid = readFileNumpy(open(datadir + valid_name).readlines(), is_debug=False)
test_rna, test_dna,test_feature, y_test = readFileNumpy(open(datadir + valid_name).readlines(), is_debug=False)

seq_size = 101
dim = 4
f_dim = 3
def build_model():
    seq_input1 = layers.Input(shape=(seq_size, dim), name='seq1')
    seq_input2 = layers.Input(shape=(seq_size, dim), name='seq2')
    l1=layers.Conv1D(hidden_dim, kernel_size)
    bn1 = layers.BatchNormalization()
    act1 = layers.ReLU()
    r1=layers.Bidirectional(layers.GRU(hidden_dim, return_sequences=True))
    l2=layers.Conv1D(hidden_dim, kernel_size)
    bn2 = layers.BatchNormalization()
    act2 = layers.ReLU()
    r2=layers.Bidirectional(layers.GRU(hidden_dim, return_sequences=True))
    l3=layers.Conv1D(hidden_dim, kernel_size)
    bn3 = layers.BatchNormalization()
    act3 = layers.ReLU()
    r3=layers.Bidirectional(layers.GRU(hidden_dim, return_sequences=True))
    l6=layers.Conv1D(hidden_dim, kernel_size)

    s1=layers.MaxPooling1D(2)(act1(bn1(l1(seq_input1))))
    s1=layers.concatenate([r1(s1), s1])
    s1=layers.MaxPooling1D(2)(act2(bn2(l2(s1))))
    s1=layers.concatenate([r2(s1), s1])
    s1=layers.MaxPooling1D(3)(act3(bn3(l3(s1))))
    s1=layers.concatenate([r3(s1), s1])
    s1=l6(s1)
    s1=layers.GlobalAveragePooling1D()(s1)

    s2=layers.MaxPooling1D(2)(act1(bn1(l1(seq_input2))))
    s2=layers.concatenate([r1(s2), s2])
    s2=layers.MaxPooling1D(2)(act2(bn2(l2(s2))))
    s2=layers.concatenate([r2(s2), s2])
    s2=layers.MaxPooling1D(3)(act3(bn3(l3(s2))))
    s2=layers.concatenate([r3(s2), s2])
    s2=l6(s2)
    s2=layers.GlobalAveragePooling1D()(s2)

    feature_input = layers.Input(shape=(f_dim), name='feat')
    fc = layers.Dense(hidden_dim)
    ac = layers.ReLU()
    feat = ac(fc(feature_input))

    merge_text = layers.concatenate([s1, s2, feat])#feat])#layers.multiply([s1, s2])

    projection = layers.Dense(128, activation='relu')
    emb = projection(merge_text)
    x = layers.Dense(128, activation='linear')(merge_text)
    emb = layers.LeakyReLU(alpha=0.3)(x)
    main_output = layers.Dense(1,activation = 'sigmoid')(emb)
    merge_model = Model(inputs=[seq_input1, seq_input2,feature_input], outputs=[main_output])
    return merge_model

def bayesian():
    seq_input1 = layers.Input(shape=(seq_size, dim), name='seq1')
    seq_input2 = layers.Input(shape=(seq_size, dim), name='seq2')
    l1=layers.Conv1D(hidden_dim, 3)
    bn1 = layers.BatchNormalization()
    act1 = layers.ReLU()
    r1=layers.Bidirectional(layers.GRU(hidden_dim, return_sequences=True))
    l2=layers.Conv1D(hidden_dim, 3)
    bn2 = layers.BatchNormalization()
    act2 = layers.ReLU()
    r2=layers.Bidirectional(layers.GRU(hidden_dim, return_sequences=True))
    l3=layers.Conv1D(hidden_dim, 3)
    bn3 = layers.BatchNormalization()
    act3 = layers.ReLU()
    r3=layers.Bidirectional(layers.GRU(hidden_dim, return_sequences=True))
    l6=layers.Conv1D(hidden_dim, 3)

    s1=layers.MaxPooling1D(2)(act1(bn1(l1(seq_input1))))
    s1=layers.concatenate([r1(s1), s1])
    s1=layers.MaxPooling1D(2)(act2(bn2(l2(s1))))
    s1=layers.concatenate([r2(s1), s1])
    s1=layers.MaxPooling1D(3)(act3(bn3(l3(s1))))
    s1=layers.concatenate([r3(s1), s1])
    s1=l6(s1)
    s1=layers.GlobalAveragePooling1D()(s1)

    s2=layers.MaxPooling1D(2)(act1(bn1(l1(seq_input2))))
    s2=layers.concatenate([r1(s2), s2])
    s2=layers.MaxPooling1D(2)(act2(bn2(l2(s2))))
    s2=layers.concatenate([r2(s2), s2])
    s2=layers.MaxPooling1D(3)(act3(bn3(l3(s2))))
    s2=layers.concatenate([r3(s2), s2])
    s2=l6(s2)
    s2=layers.GlobalAveragePooling1D()(s2)

    feature_input = layers.Input(shape=(f_dim), name='feat')
    fc = layers.Dense(hidden_dim)
    ac = layers.ReLU()
    feat = ac(fc(feature_input))

    # first part
    merge_text = layers.concatenate([s1, s2])  # P(I|seq)
    #,feat])#layers.multiply([s1, s2])
    projection = layers.Dense(hidden_dim,activation='relu')
    emb = projection(merge_text)
    x = layers.Dense(hidden_dim, activation='linear')(emb)
    emb = layers.LeakyReLU(alpha=0.3)(x) + x
    main_output1 = layers.Dense(1,activation = 'sigmoid')(emb)  # p(I|S,H,O)

    # second part
    #,feat])#layers.multiply([s1, s2])
    projection2 = layers.Dense(hidden_dim,activation='relu')
    emb2 = projection2(feat) + feat
    x2 = layers.Dense(hidden_dim, activation='linear')(emb2)
    emb2 = layers.LeakyReLU(alpha=0.3)(x2) + x2
    main_output2 = layers.Dense(1,activation = 'sigmoid')(emb2)  # p(I|S,H,O)

    final_predict = layers.concatenate([emb, emb2])
    final_predict = layers.Dense(hidden_dim,activation = 'sigmoid')(final_predict)
    final_predict = layers.Dense(1,activation = 'sigmoid')(final_predict)

    merge_model = Model(inputs=[seq_input1, seq_input2,feature_input], outputs=[final_predict, main_output1, main_output2 ])
    return merge_model
def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy


batch_size1 = 32
if model_name=='simple':
    model = build_model()
else:
    model = bayesian()
adam = Adam(lr=1e-4, amsgrad=True, epsilon=1e-5)
model.compile(
    loss='binary_crossentropy',
    metrics=["accuracy"],
    optimizer="adam",
)
model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    f'../checkpoint/bayesian/{exp}_best.hdf5',
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    options=None,
)
es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

def sigmoid(x):
    x = np.array(x)

    sig = 1 / (1 + np.exp(-x))
    return sig

if args.mode == 'train':
    if model_name=='simple':
        model.fit([train_rna,train_dna,train_feature], y_train, validation_data=([valid_rna, valid_dna,valid_feature],  y_valid),
            epochs=20,
            callbacks = [checkpoint, es],
            batch_size = 32
            )
    else:
        model.fit([train_rna,train_dna,train_feature], [y_train, y_train, y_train], validation_data=([valid_rna, valid_dna,valid_feature], [y_valid, y_valid, y_valid,]),
                epochs=10,
                callbacks = [checkpoint, es],
                batch_size = 32
                )
model.load_weights(f'../checkpoint/{exp}_best.hdf5')
pred = model.predict([test_rna, test_dna, test_feature])
if not os.path.isdir('result'):
    os.system('mkdir result')
outputdir=f'result/'

pkl.dump([pred, y_test],open(f'{outputdir}/{exp}_best.bin','wb'))

def get_metrics(preds, labels):
    acc = accuracy_score(labels, preds>0.5)
    sn = precision_score(labels, preds>0.5)
    sp = recall_score(labels, preds>0.5)
    auroc = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)
    f1 = f1_score(labels, preds>0.5)
    return [acc, sn, sp, auroc, auprc, f1]
if model_name=='simple':
    print(get_metrics(pred,y_test))
else:
    print(get_metrics(pred[0],y_test))
