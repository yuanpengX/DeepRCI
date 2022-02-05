import numpy as np
from const import *
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, accuracy_score
import torch
import random

def seed_all(seed):
    if not seed:
        seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def posMask(length):
    return np.array([
        True if i > length else False for i in range(1, input_length + 1)
    ]).reshape(-1)


def positionalEncoding(length):

    process_func = [absPosition, relaPosition, oriPe]
    pe = np.zeros((input_length, len(process_func)))
    valid_pe = np.array([f(length) for f in process_func]).T
    pe[:length] = valid_pe
    return pe


def get_metrics(pred, label):
    auroc = roc_auc_score(label, pred)
    auprc = average_precision_score(label, pred)
    pred_binary = np.array(pred) > 0.5
    f1 = f1_score(label, pred_binary)
    acc = f1_score(label, pred_binary)
    metric_str = "auroc %.3f auprc %.3f f1: %.3f acc: %.3f" % (auroc, auprc,
                                                               f1, acc)
    return metric_str


def stringOnehot(line):
    x = [dic[ch] for ch in line]
    return np.array(x)


def readFileNumpy(sample_name, is_debug=True):
    data_np = None
    with open(sample_name, 'r') as fp:
        all_lines = fp.readlines()
        data_size = debug_size if is_debug else len(all_lines)
        data_np = np.zeros((data_size, input_length * 2, 4))
        labels = [
            0,
        ] * data_size
        for index, line in enumerate(all_lines):
            if is_debug and index >= data_size:
                break
            rna, dna, label = line.strip().split()

            rna = rna[:input_length] + "N" * max(input_length - len(rna), 0)
            dna = dna[:input_length] + "N" * max(input_length - len(dna), 0)
            data = stringOnehot(rna + dna)
            data_np[index] = data
            labels[index] = float(label)
        return data_np, labels


if __name__ == "__main__":
    print(positionalEncoding(10))
