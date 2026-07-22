from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model import GCN
from sklearn.metrics import f1_score
from params_M import *
import warnings
import os



warnings.filterwarnings('ignore')
dataset = 'pubmed'
args = set_params(dataset)

dataset_str = args.dataset

adj, features, labels, idx_train, idx_val, idx_test = load_data2(dataset_str, if_feat=False)
graph_test = ['none', '_add_0.5', '_add_0.75', '_dele_0.5', '_dele_0.2']
for i in graph_test:
    if i == 'none':
        adj_add = adj
    else:
        adj_add = sp.load_npz(f"../data/{dataset_str}-graph/{dataset_str}{i}.npz")
        adj_add = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj_add))
        adj_add = adj_add.to_dense()

    adj_hot = torch.where(adj_add > 0, 1, 0)
    print(i, "degree:", adj_hot.sum(1).float().mean())
