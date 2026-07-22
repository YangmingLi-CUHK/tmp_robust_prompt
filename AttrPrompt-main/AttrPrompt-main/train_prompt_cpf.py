from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model_prompt1 import Node_prompt, Vector_prompt, CLUB
from sklearn.metrics import f1_score
from params_M import *
import warnings
import os
import random


warnings.filterwarnings('ignore')

args = set_params()
print(args)
args.IB = (args.IB == 'IB')
dataset_str = args.dataset
k = args.k
from utils_multi import evaluate_whole_graph, eval_rocauc

if dataset_str in ['cora', 'citeseer']:
    adj, adj_f, features, labels, idx_train, idx_val, idx_test = load_data2(dataset_str, if_feat=True, k=k)
    root_path = 'save1/GCN/{}'.format(dataset_str)
    ib_norm = True
# if dataset_str in ['pubmed', 'cora']:
#    ib_norm = False
elif dataset_str in ['A-photo', 'A-computer']:
    adj, adj_f, features, labels, idx_train, idx_val, idx_test = load_cpf_data(dataset_str, seed=0, labelrate_train=20, labelrate_val=30, if_feat=True, k=k)
    root_path = 'save/GCN/{}'.format(dataset_str)
    ib_norm = False
else:
    raise ValueError('Invalid dataname')


if not os.path.exists(root_path):
    raise ValueError('Pretrain First')

acc_all, f1_all = [], []
for seed in range(10):
    random.seed(seed)                       # Python 随机种子
    np.random.seed(seed)                    # NumPy 随机种子
    torch.manual_seed(seed)                 # CPU 上的 PyTorch 随机种子
    torch.cuda.manual_seed(seed)            # 当前 GPU 的 PyTorch 随机种子
    torch.cuda.manual_seed_all(seed)        # 所有 GPU 的 PyTorch 随机种子（多卡）

    save_path = os.path.join(root_path, "model_{}.pth".format(seed))
    if dataset_str in ['cora', 'citeseer']:
        hidden = 16
        epochs = args.epochs
        lr = 0.01
        weight_decay = 5e-4
        
    elif dataset_str in ['A-photo', 'A-computer']:
        hidden = 64
        epochs = args.epochs
        lr = 0.001
        weight_decay = 0.0001

    else:
        raise ValueError('wrong dataset name')



    def train(epoch, alpha, warm_up, club_epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        if model.IB:
            output, out_teacher, x_prompt, H_noisy = model(features, adj, adj_f, alpha,
                                                                  dataset_str=args.dataset, train=True)
            loss_ib = club_module(x_prompt, H_noisy)
        else:
            output, out_teacher = model(features, adj, adj_f, alpha=alpha, dataset_str=args.dataset, train=True)
        # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_kl = F.kl_div(output, out_teacher[0], reduction='batchmean', log_target=True)
        if not model.IB:
            loss_train = loss_kl
            loss_train.backward()
            optimizer.step()
            #print("Epoch: {:04d}, loss_kl: {:.4f}".format(epoch, loss_kl.item()))
            return loss_train
        else:
            if epoch < warm_up:

                loss_club_train = club_module.learning_loss(x_prompt.detach(), H_noisy.detach())
                optimizer_club.zero_grad()
                loss_club_train.backward()
                optimizer_club.step()
                #print(
                #    "Epoch: {:04d}, loss_club_train: {:.4f}".format(
                #        epoch, loss_club_train.item()))
                return 999-epoch
            else:
                # todo
                loss_train = loss_kl + args.w_ib*loss_ib
                loss_train.backward()
                optimizer.step()

                loss_club_train = torch.tensor([0])
                for i in range(club_epoch):
                # if epoch % 5 == 0:
                    loss_club_train = club_module.learning_loss(x_prompt.detach(), H_noisy.detach())
                    optimizer_club.zero_grad()
                    loss_club_train.backward()
                    optimizer_club.step()
                #print("Epoch: {:04d}, loss_train: {:.4f}, loss_kl: {:.4f}, loss_ib: {:.4f}, loss_club_train: {:.4f}".format(
                #    epoch, loss_train.item(), loss_kl.item(), loss_ib.item(), loss_club_train.item()))
                return loss_train

        # acc_train = accuracy(output[idx_train], labels[idx_train])


        # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        # acc_val = accuracy(output[idx_val], labels[idx_val])

        # print('Epoch: {:04d}'.format(epoch+1),
        #       'loss_train: {:.4f}'.format(loss_train.item()),
        #       'acc_train: {:.4f}'.format(acc_train.item()),
        #       'loss_val: {:.4f}'.format(loss_val.item()),
        #       'acc_val: {:.4f}'.format(acc_val.item()),
        #       'time: {:.4f}s'.format(time.time() - t))


    def average_cosine_similarity(tensor1, tensor2):
        # tensor1, tensor2: [N, F]
        sim = F.cosine_similarity(tensor1, tensor2, dim=1)  # [N]
        return sim.mean().item()

    def rowwise_euclidean_distance(A, B):
        return torch.norm(A - B, dim=1).mean()



                
    def test(alpha):
        model.eval()
        graph_test = ['_add_0.5', '_add_0.75', '_dele_0.5', '_dele_0.2']
        result_acc = []
        result_f1 = []
        for i in graph_test:
            adj_add = sp.load_npz(f"../data/{dataset_str}-graph/{dataset_str}{i}.npz")
            adj_add = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj_add))
            adj_add = adj_add.cuda(args.cuda_id)
            output = model(features, adj_add, adj_f, alpha)
            acc_test = accuracy(output[idx_test], labels[idx_test])
            f1 = F1score(output[idx_test].cpu(),labels[idx_test].cpu())
            result_acc.append(acc_test.item())
            result_f1.append(f1.item())
            print(i, "accuracy= {:.4f}".format(acc_test.item()), "f1-score={:.4f}".format(f1.item()),'\n')
        return result_acc, result_f1


    model_type = "Vector_"
    if model_type == "Vector":
        model = Vector_prompt(nfeat=features.shape[1],
                            nhid=hidden,
                            nclass=labels.max().item() + 1,
                            dropout=args.dropout,
                            p=args.p,
                            cuda_id=args.cuda_id,
                            num_nodes=features.shape[0]
                              )
    else:
        model = Node_prompt(nfeat=features.shape[1],
                    nhid=hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout,
                    p=args.p,
                    IB=args.IB,
                    attack_iters=args.attack_iters,
                    step_size=args.step_size,
                    mask_type=args.mask_type,
                    prompt_type=args.prompt_type,
                    norm_if = ib_norm,
                    cuda_id=args.cuda_id,
                    lp=args.lp,
                    num_nodes=features.shape[0]
                    )
        if model.IB:
            club_module = CLUB(features.shape[1], labels.max().item() + 1, args.hclub, args.lclub)
            club_module = club_module.cuda(args.cuda_id)
            optimizer_club = torch.optim.Adam(club_module.parameters(), lr=args.club_opt_lr)


    if os.path.exists(save_path):
        state_dict = torch.load(save_path)
        model.model_teacher.load_state_dict(state_dict)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)
    cuda = torch.cuda.is_available()

    if cuda:
        model.cuda(args.cuda_id)
        features = features.cuda(args.cuda_id)
        labels = labels.cuda(args.cuda_id)
        adj = adj.cuda(args.cuda_id)
        adj_f = adj_f.cuda(args.cuda_id)
        if args.dataset != 'twitch-e':
            idx_train = idx_train.cuda(args.cuda_id)
            idx_val = idx_val.cuda(args.cuda_id)
            idx_test = idx_test.cuda(args.cuda_id)
    if model.generator.type_p == "GAT":
        adj_f = torch.where(adj_f>0, 1, 0)
    # Train model
    t_total = time.time()

    best_val = 0
    best_epoch = 0
    patient = 0
    best_loss = 1000
    for epoch in range(epochs):
        loss_train = train(epoch, args.alpha, args.warm_up, args.club_epoch)
    acc_list, f1_list = test(args.alpha)
    acc_all.append(acc_list)
    f1_all.append(f1_list)

acc_all = np.array(acc_all)
f1_all = np.array(f1_all)

graph_test = ['_add_0.5', '_add_0.75', '_dele_0.5', '_dele_0.2']
acc_all = acc_all * 100
f1_all = f1_all * 100
acc_mean = acc_all.mean(0)
f1_mean = f1_all.mean(0)
acc_var = acc_all.var(0)
f1_var = f1_all.var(0)

print(acc_mean)

print(acc_all)
print(f1_all)
print("GCN with prompt {} with load mask".format(dataset_str, args.alpha))
for n, acc, f1, acc_var, f1_var in zip(graph_test, acc_mean, f1_mean, acc_var, f1_var):
    print(n, "accuracy_mean= {:.4f}+-{:.4f}".format(acc, acc_var), "f1-score_mean={:.4f}+-{:.4f}".format(f1, f1_var))
print(args)


