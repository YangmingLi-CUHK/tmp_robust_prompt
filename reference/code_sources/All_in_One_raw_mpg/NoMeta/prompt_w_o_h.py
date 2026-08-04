import torch
import pickle as pk
from random import shuffle
from utils import seed_everything, seed

seed_everything(seed)
from torch_geometric.data import Batch, Data
from MPG.Models import GCN
from torch import nn, optim
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from collections import defaultdict
import numpy as np
from MPG.utils import Evaluator
import warnings


class PromptGraph(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_class, class_num, inner_prune=None):
        """
        :param token_dim:
        :param token_num:
        :param prune_thre: if inner_prune is None, then all inner and cross prune will adopt this prune_thre
        :param isolate_tokens: if Trure, then inner tokens have no connection.
        :param inner_prune: if inner_prune is not None, then cross prune adopt prune_thre whereas inner prune adopt inner_prune
        """
        super(PromptGraph, self).__init__()

        self.inner_prune = inner_prune

        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_class, token_dim)) for i in range(class_num)])

        for token in self.token_list:
            torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)

    def token_view(self, ):
        pg_list = []
        for i, tokens in enumerate(self.token_list):
            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1

            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            edge_index = inner_adj.nonzero().t().contiguous()

            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))

        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch


def multi_class_data(dataname, num_class):
    statistic = defaultdict(list)

    # load training NIG (node induced graphs)
    train_list = []
    for task_id in range(num_class):
        data_path1 = '../Dataset/{}/induced_graphs/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = '../Dataset/{}/induced_graphs/task{}.meta.train.query'.format(dataname, task_id)

        with (open(data_path1, 'br') as f1, open(data_path2, 'br') as f2):
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:100]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                train_list.append(g)

    shuffle(train_list)
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in range(num_class):
        data_path1 = '../Dataset/{}/induced_graphs/task{}.meta.test.support'.format(dataname, task_id)
        data_path2 = '../Dataset/{}/induced_graphs/task{}.meta.test.query'.format(dataname, task_id)

        with (open(data_path1, 'br') as f1, open(data_path2, 'br') as f2):
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:100]

            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                test_list.append(g)

    shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        print(key, value)

    return train_data, test_data


def mrr_hit(normal_label: np.ndarray, pos_out: np.ndarray, metric: list = None):
    if isinstance(normal_label, np.ndarray) and isinstance(pos_out, np.ndarray):
        pass
    else:
        warnings.warn('it would be better if normal_label and out are all set as np.ndarray')

    results = {}
    if not metric:
        metric = ['mrr', 'hits']

    if 'hits' in metric:
        hits_evaluator = Evaluator(eval_metric='hits@50')
        flag = normal_label
        pos_test_pred = torch.from_numpy(pos_out[flag == 1])
        neg_test_pred = torch.from_numpy(pos_out[flag == 0])

        for N in [100]:
            neg_test_pred_N = neg_test_pred.view(-1, 100)
            for K in [1, 5, 10]:
                hits_evaluator.K = K
                test_hits = hits_evaluator.eval({
                    'y_pred_pos': pos_test_pred,
                    'y_pred_neg': neg_test_pred_N,
                })[f'hits@{K}']

                results[f'Hits@{K}@{N}'] = test_hits

    if 'mrr' in metric:
        mrr_evaluator = Evaluator(eval_metric='mrr')
        flag = normal_label
        pos_test_pred = torch.from_numpy(pos_out[flag == 1])
        neg_test_pred = torch.from_numpy(pos_out[flag == 0])

        neg_test_pred = neg_test_pred.view(-1, 100)

        mrr = mrr_evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })

        if isinstance(mrr, torch.Tensor):
            mrr = mrr.item()
        results['mrr'] = mrr
    return results


def eva(pre, label, task_type='multi_class_classification'):
    if task_type == 'regression':
        mae = mean_absolute_error(label, pre)
        mse = mean_squared_error(label, pre)
        return {"mae": mae, "mse": mse}
    elif task_type == 'multi_class_classification':
        pre_cla = torch.argmax(pre, dim=1)
        acc = accuracy_score(label, pre_cla)
        mac_f1 = f1_score(label, pre_cla, average='macro')
        mic_f1 = f1_score(label, pre_cla, average='micro')
        return {"acc": acc, "mac_f1": mac_f1, "mic_f1": mic_f1}
    elif task_type == 'link_prediction':
        normal_label = label
        pos_out = pre[:, 1]
        results = mrr_hit(normal_label, pos_out)
        return results
    else:
        raise NotImplemented(
            "eva() function is currently only used for multi-class classification  and link_prediction tasks!")


def model_create(dataname, gnn_type, num_class, task_type='multi_class_classification'):
    input_dim, hid_dim = 100, 100
    lr, wd = 0.001, 0.00001
    tnpc = 100

    gnn = GCN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=2, gnn_type=gnn_type)
    pre_train_path = './pre_trained_gnn/{}.GraphCL.{}.pth'.format(dataname, gnn_type)
    gnn.load_state_dict(torch.load(pre_train_path))
    print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
    for p in gnn.parameters():
        p.requires_grad = False

    PG = PromptGraph(token_dim=input_dim, token_num_per_class=tnpc, class_num=num_class, inner_prune=0.01)

    opi = optim.Adam(filter(lambda p: p.requires_grad, PG.parameters()), lr=lr,
                     weight_decay=wd)

    if task_type == 'regression':
        lossfn = nn.MSELoss(reduction='mean')
    else:
        lossfn = nn.CrossEntropyLoss(reduction='mean')

    return gnn, PG, opi, lossfn


def testing(test, PG, gnn, task_type='multi_class_classification'):
    PG.eval()
    emb0 = gnn(test.x, test.edge_index, test.batch)
    pg_batch = PG.token_view()
    pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
    dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))

    if task_type == 'multi_class_classification':
        pre = torch.softmax(dot, dim=1)
    elif task_type == 'regression':
        pre = torch.sigmoid(dot)
        pre = pre.detach()

    res = eva(pre, test.y, task_type=task_type)
    return res


if __name__ == '__main__':

    dataname, gnn_type, num_class = "CiteSeer", "TransformerConv", 6
    train, test = multi_class_data(dataname, num_class)
    task_type = 'multi_class_classification'
    gnn, PG, opi, lossfn = model_create(dataname, gnn_type, num_class, task_type)
    prompt_epoch = 200
    # training stage
    PG.train()
    emb0 = gnn(train.x, train.edge_index, train.batch)
    for j in range(prompt_epoch):
        pg_batch = PG.token_view()
        pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)

        dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
        if task_type == 'multi_class_classification':
            sim = torch.softmax(dot, dim=1)
        elif task_type == 'regression':
            sim = torch.sigmoid(dot)  # 0-1
        else:
            raise KeyError("task type error!")

        train_loss = lossfn(sim, train.y)

        print('{}/{} training loss: {:.8f}'.format(j, prompt_epoch, train_loss.item()))

        opi.zero_grad()
        train_loss.backward()
        opi.step()

        if j % 5 == 0:
            res = testing(test, PG, gnn, task_type=task_type)
            if task_type == 'regression':
                print("""MAE: {:.4} | MSE: {:.4} """.format(res["mae"], res["mse"]))

            else:
                print("""Acc: {:.4} | Macro F1: {:.4} | Micro F1: {:.4}""".format(res["acc"],
                                                                                  res["mac_f1"],
                                                                                  res["mic_f1"]))
            PG.train()
