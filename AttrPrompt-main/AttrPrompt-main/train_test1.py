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
import random

warnings.filterwarnings('ignore')
dataset = 'pubmed'
args = set_params(dataset)

dataset_str = args.dataset

root_path = 'save/GCN/{}'.format(dataset_str)
if not os.path.exists(root_path):
    os.makedirs(root_path)


adj, features, labels, idx_train, idx_val, idx_test = load_data2(dataset_str, if_feat=False)

acc_all, f1_all = [], []
for seed in range(10):
    # random.seed(seed)                       # Python 随机种子
    # np.random.seed(seed)                    # NumPy 随机种子
    # torch.manual_seed(seed)                 # CPU 上的 PyTorch 随机种子
    # torch.cuda.manual_seed(seed)            # 当前 GPU 的 PyTorch 随机种子
    # torch.cuda.manual_seed_all(seed)        # 所有 GPU 的 PyTorch 随机种子（多卡）

    save_path = os.path.join(root_path, "model_{}.pth".format(seed))
    hidden = 16
    epochs = 400
    lr = 0.01
    weight_decay = 5e-4

    def test():
        result_acc = []
        result_f1 = []
        model.eval()
        graph_test = ['none', '_add_0.5', '_add_0.75', '_dele_0.5', '_dele_0.2']
        for i in graph_test:
            if i == 'none':
                adj_add = adj
            else:
                adj_add = sp.load_npz(f"../data/{dataset_str}-graph/{dataset_str}{i}.npz")
                adj_add = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj_add))
                adj_add = adj_add.cuda(args.cuda_id)
            output = model(features,adj_add,0,test = 1)
            acc_test = accuracy(output[idx_test], labels[idx_test])
            f1 = F1score(output[idx_test].cpu(),labels[idx_test].cpu())
            result_acc.append(acc_test.item())
            result_f1.append(f1.item())
            print(i, "accuracy= {:.4f}".format(acc_test.item()), "f1-score={:.4f}".format(f1.item()), '\n')
        return result_acc, result_f1



    model = GCN(nfeat=features.shape[1],
                nhid=hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,
                p=args.p,
                cuda_id=args.cuda_id)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)
    cuda = torch.cuda.is_available()

    if cuda:
        model.cuda(args.cuda_id)
        features = features.cuda(args.cuda_id)
        labels = labels.cuda(args.cuda_id)
        adj = adj.cuda(args.cuda_id)
        idx_train = idx_train.cuda(args.cuda_id)
        idx_val = idx_val.cuda(args.cuda_id)
        idx_test = idx_test.cuda(args.cuda_id)

    torch.autograd.set_detect_anomaly(True)
    # Train model
    t_total = time.time()

    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict)


    acc_list, f1_list = test()
    acc_all.append(acc_list)
    f1_all.append(f1_list)

acc_all = np.array(acc_all)
f1_all = np.array(f1_all)

graph_test = ['none', '_add_0.5', '_add_0.75', '_dele_0.5', '_dele_0.2']

acc_all = acc_all * 100
f1_all = f1_all * 100
acc_mean = acc_all.mean(0)
f1_mean = f1_all.mean(0)
acc_var = acc_all.var(0)
f1_var = f1_all.var(0)
print(acc_all)
print(f1_all)
print("GCN on {}".format(dataset_str))
for n, acc, f1, acc_var, f1_var in zip(graph_test, acc_mean, f1_mean, acc_var, f1_var):
    print(n, "accuracy_mean= {:.4f}+-{:.4f}".format(acc, acc_var), "f1-score_mean={:.4f}+-{:.4f}".format(f1, f1_var))


