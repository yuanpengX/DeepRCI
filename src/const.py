dic = {
    'A': [1, 0, 0, 0],
    'T': [0, 1, 0, 0],
    'C': [0, 0, 1, 0],
    'G': [0, 0, 0, 1],
    'R': [0.5, 0, 0, 0.5],
    'Y': [0, 0.5, 0.5, 0, 0],
    'M': [0.5, 0, 0.5, 0],
    'K': [0, 0.5, 0, 0.5],
    'S': [0, 0, 0.5, 0.5],
    'W': [0.5, 0.5, 0, 0],
    'H': [0.33, 0.33, 0.33, 0],
    'B': [0, 0.33, 0.33, 0.33],
    'V': [0.33, 0, 0.33, 0.33],
    'D': [0.33, 0.33, 0, 0.33],
    'N': [0.25, 0.25, 0.25, 0.25],
}

# data related
input_length = 101
datadir = 'iMAGIC/data/dataset/'

train_name = 'debug_inter_train.data'
valid_name = 'debug_inter_valid.data'
is_debug = True
debug_size = 10000

# model related
d_model = 1024
nhead = 8
num_layers = 6
hidden = 128
kernel_size = 3

# training related
max_epoch = 2000
step_size = 10

seed=0
weight_decay = 5e-4
lr = 1e-4
batch_size = 1