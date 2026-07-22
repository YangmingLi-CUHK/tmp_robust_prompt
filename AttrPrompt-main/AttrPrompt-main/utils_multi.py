import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score

@torch.no_grad()

def evaluate_whole_graph(model, dataset_tr, dataset_val, datasets_te, eval_func, data_loaders=None, neighbor=False,
                         cuda_id=0, prompt=False):
    model.eval()
    if neighbor:
        loader_tr, loader_val, loader_te = data_loaders[0], data_loaders[1], data_loaders[2]
        train_out = model.inference(dataset_tr, loader_tr)
        train_acc = eval_func(dataset_tr.label, train_out)
        valid_out = model.inference(dataset_val, loader_val)
        valid_acc = eval_func(dataset_val.label, valid_out)
        test_outs, test_accs = [], []
        for dataset_te in datasets_te:
            test_out = model.inference(dataset_te, loader_te)
            test_outs.append(test_out)
            test_accs.append(eval_func(dataset_te.label, test_out))
    else:
        if not prompt:
            accs, test_outs = [], []
            train_out = model.inference(dataset_tr)
            train_acc = eval_func(dataset_tr.label, train_out)
            valid_out = model.inference(dataset_val)
            valid_acc = eval_func(dataset_val.label, valid_out)
            accs += [train_acc] + [valid_acc]
            for i, dataset in enumerate(datasets_te):
                out = model.inference(dataset)
                test_outs.append(out)
                accs.append(eval_func(dataset.label, out))
        else:
            accs, test_outs = [], []
            train_out = model.model_teacher.inference(dataset_tr)
            train_acc = eval_func(dataset_tr.label, train_out)
            # val
            x, y = dataset_val.graph['node_feat'].to(model.device), dataset_val.label.to(model.device)
            adj_f = dataset_val.graph['adj_f'].to(model.device)
            adj_a = dataset_val.graph['adj'].to(model.device)
            val_out = model(x, adj_a, adj_f, train=False)
            valid_acc = eval_func(dataset_val.label, val_out)
            accs += [train_acc] + [valid_acc]
            # test
            for i, dataset in enumerate(datasets_te):
                x, y = dataset.graph['node_feat'].to(model.device), dataset.label.to(model.device)
                adj_f = dataset.graph['adj_f'].to(model.device)
                adj_a = dataset.graph['adj'].to(model.device)
                out = model(x, adj_a, adj_f, train=False)
                test_outs.append(out)
                accs.append(eval_func(dataset.label, out))
    return accs, test_outs


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)

