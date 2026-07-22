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
from utils_multi import *

warnings.filterwarnings('ignore')
dataset = 'twitch-e'
args = set_params(dataset)

dataset_str = args.dataset

# i = '_add_0.5'
# adj_add = sp.load_npz(f"../data/{dataset_str}-graph/{dataset_str}{i}.npz")

root_path = 'save/GCN/{}'.format(dataset_str)
if not os.path.exists(root_path):
    os.makedirs(root_path)

if dataset_str in ['cora', 'citeseer']:
    adj, features, labels, idx_train, idx_val, idx_test = load_data2(dataset_str, if_feat=False)
if dataset_str in ['A-photo', 'A-computer']:
    adj, features, labels, idx_train, idx_val, idx_test = load_cpf_data(dataset_str, seed=0, labelrate_train=20, labelrate_val=30, if_feat=False)

else:
    raise ValueError('Invalid dataname')
# print(f"Train num nodes {dataset_tr.n} | num classes {dataset_tr.c} | num node feats {dataset_tr.d}")
# print(f"Val num nodes {dataset_val.n} | num classes {dataset_val.c} | num node feats {dataset_val.d}")
# for i in range(len(te_subs)):
#     dataset_te = datasets_te[i]
#     print(f"Test {i} num nodes {dataset_te.n} | num classes {dataset_te.c} | num node feats {dataset_te.d}")

acc_all, f1_all = [], []
for seed in range(10):
    random.seed(seed)                       # Python 随机种子
    np.random.seed(seed)                    # NumPy 随机种子
    torch.manual_seed(seed)                 # CPU 上的 PyTorch 随机种子
    torch.cuda.manual_seed(seed)            # 当前 GPU 的 PyTorch 随机种子
    torch.cuda.manual_seed_all(seed)        # 所有 GPU 的 PyTorch 随机种子（多卡）

    save_path = os.path.join(root_path, "model_{}.pth".format(seed))
    if dataset_str in ['cora', 'citeseer', 'pubmed']:
        hidden = 16
        epochs = args.epochs
        lr = 0.01
        weight_decay = 5e-4

    elif dataset_str in ['A-photo', 'A-computer']:
        hidden = 64
        epochs = args.epochs
        lr = 0.001
        weight_decay = 0.0001
    elif dataset_str in ['twitch-e']:
        hidden = 32
        lr = 0.01
        epochs = 200
        weight_decay = 1e-3
    else:
        raise ValueError('Invalid para')

    def train_multi(epoch, criterion, best_test, best_acc_val):
        best_test = best_test
        save_model = False
        t = time.time()
        model.train()
        optimizer.zero_grad()
        y = labels
        pred = model(features, adj, softmax_chose=False)
        if y.shape[1] == 1:
            true_label = F.one_hot(y.long(), y.max() + 1).squeeze(1)
        else:
            true_label = y
        loss_train = criterion(pred, true_label.squeeze(1).to(torch.float))
        loss_train.backward()
        optimizer.step()

        accs, test_outs = evaluate_whole_graph(model, dataset_tr, dataset_val, datasets_te, eval_rocauc, args.cuda_id)
        if best_acc_val < accs[1]:
            best_acc_val = accs[1]
            best_test = accs
            save_model = True

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'Train: {:.2f}%, '.format(100 * accs[0]),
              'Valid: {:.2f}%, '.format(100 * accs[1]),
              'time: {:.4f}s'.format(time.time() - t))

        test_info = ''
        for test_acc in accs[2:]:
            test_info += f'Test: {100 * test_acc:.2f}% '
        print(test_info)
        return best_test, best_acc_val, save_model

    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj,epoch,test = 0)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))


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


    if args.dataset == 'twitch-e':
        nclass = nclass
    else:
        nclass = labels.max().item() + 1
    model = GCN(nfeat=features.shape[1],
                nhid=hidden,
                nclass=nclass,
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
        if args.dataset != 'twitch-e':
            idx_train = idx_train.cuda(args.cuda_id)
            idx_val = idx_val.cuda(args.cuda_id)
            idx_test = idx_test.cuda(args.cuda_id)

    torch.autograd.set_detect_anomaly(True)
    # Train model
    t_total = time.time()
    if args.dataset != 'twitch-e':
        for epoch in range(epochs):
            train(epoch)
        torch.save(model.state_dict(), save_path)
        print("sucess save model")

        acc_list, f1_list = test()
        acc_all.append(acc_list)
        f1_all.append(f1_list)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
        best_acc_val = 0
        accs = 0
        for epoch in range(epochs):
            accs, best_acc_val, save_model = train_multi(epoch, criterion, accs, best_acc_val)
            if save_model:
                torch.save(model.state_dict(), save_path)
        print("sucess save model")
        acc_all.append(accs)

if args.dataset != 'twitch-e':
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
else:
    acc_all = np.array(acc_all)

    graph_test = ['ES', 'FR', 'PTBR', 'RU', 'TW']

    acc_all = acc_all * 100
    acc_mean = acc_all.mean(0)
    acc_var = acc_all.var(0)
    print(acc_all)
    print("GCN on {}".format(dataset_str))

    print("Train: accuracy_mean= {:.4f}+-{:.4f}".format(acc_mean[0], acc_var[0]))
    print("Val: accuracy_mean= {:.4f}+-{:.4f}".format(acc_mean[1], acc_var[1]))
    for n, acc, acc_var, in zip(graph_test, acc_mean[2:], acc_var[2:]):
        print(n, "accuracy_mean= {:.4f}+-{:.4f}".format(acc, acc_var))


