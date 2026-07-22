import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch_geometric.nn import dense_mincut_pool
from layers import GraphConvolution
from utils import *
from layers import GraphAttentionLayer
from torch import Tensor
from torch_geometric.nn.inits import glorot

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, p, cuda_id):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.p = p

        self.device = torch.device(('cuda:' + str(cuda_id)) if torch.cuda.is_available() else 'cpu')

    def forward(self, x, adj, epoch=0, test=0, return_x=False, softmax_chose=True):

        x = F.relu(self.gc1(x, adj))


        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x_rep1 = x
        if softmax_chose:
            out = F.log_softmax(x, dim=1)
        else:
            out = x
        if return_x:
            return out, x_rep1
        else:
            return out


        # x_rep2 = x
        # if return_x:
        #     return F.log_softmax(x, dim=1), x_rep1, x_rep2
        # else:
        #     return F.log_softmax(x, dim=1)

    def inference(self, data, feat=None):
        x, y = data.graph['node_feat'].to(self.device), data.label.to(self.device)
        if feat != None:
            x = feat
        adj = data.graph['adj'].to(self.device)
        pred = self.forward(x, adj, softmax_chose=False)
        return pred

    def encode_with_prompt(self, x, adj, x_promt1, x_promt2, alpha):
        x = alpha * x + (1 - alpha) * x_promt1
        x = F.relu(self.gc1(x, adj))
        x = alpha * x + alpha * x_promt2
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def encoder_layer1(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def encoder_layer2(self, x, adj):
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class DynamicPrompt(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, dropout, p, cuda_id, norm_if=False, lp=0.2):
        super(DynamicPrompt, self).__init__()
        self.type_p = "GCN"
        if self.type_p == "GAT":
            self.prompt_g = GraphAttentionLayer(nfeat, nhid, dropout=dropout, concat=True)
        else:
            self.prompt_g = GraphConvolution(nfeat, nhid)
        # self.prompt_g2 = GraphConvolution(nfeat, nhid)

        # self.prompt_g2 = GraphConvolution(nfeat, nhid)

        self.norm_if = norm_if

        self.beta = not self.norm_if

        if not self.norm_if:
            self.gamma_1 = nn.Sequential(
                nn.Linear(nfeat, nfeat, bias=False),
                # nn.ReLU(),
                # nn.Linear(64, nfeat),
                # nn.Sigmoid()
            )
        else:
            self.gamma_1 = nn.Sequential(
                nn.Linear(nfeat, nfeat, bias=False),
            )

        # self.gamma_1 = nn.Linear(nfeat, nfeat, bias=False)
        # self.gamma_2 = nn.Linear(nfeat, nfeat, bias=False)
        if self.beta:
            self.beta_1 = nn.Linear(nfeat, nfeat, bias=False)
        # self.beta_2 = nn.Linear(nfeat, nfeat, bias=False)

        self.elu = nn.ELU()
        self.lrelu = nn.LeakyReLU(lp)

        self.device = torch.device(('cuda:' + str(cuda_id)) if torch.cuda.is_available() else 'cpu')

    def forward(self, x, adj_f, adj_a, att=False):
        base_prompt = F.relu(self.prompt_g(x, adj_f))  # 先学一个固定的prompt
        # base_prompt = F.normalize(F.relu(self.prompt_g(x, adj_f)), dim=1)
        # base_prompt = F.relu(self.prompt_g2(base_prompt, adj_f))  #先学一个固定的prompt
        if adj_a.is_sparse:
            adj_a = adj_a.to_dense()
        adj_a.fill_diagonal_(0)
        neighbor = torch.mm(adj_a, x)
        # neighbor = 0.5*(neighbor+x)
        #
        gamma = self.gamma_1(neighbor)  # + self.gamma_2(x)
        if not self.norm_if:
            gamma = self.lrelu(gamma) + 1
        else:
            gamma = self.lrelu(gamma) + 1
        learned_prompt = gamma * base_prompt

        if self.beta:
            beta = self.beta_1(neighbor)  # + self.beta_2(x)
            beta = self.lrelu(beta)
            learned_prompt = gamma * base_prompt + beta
        if not self.norm_if:
            learned_prompt = F.normalize(learned_prompt)
        else:
            learned_prompt = learned_prompt

        if att:
            return learned_prompt, gamma
        else:
            return learned_prompt




class CLUB(nn.Module):
    def __init__(self, dim_P, dim_H, hclub, hlayer):
        super().__init__()
        self.mu_net = self._build_mlp(dim_P, dim_H, hclub, hlayer)
        self.logvar_net = self._build_mlp(dim_P, dim_H, hclub, hlayer)

    def _build_mlp(self, input_dim, output_dim, hidden_dim, num_layers):
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, P, H):
        # Estimate MI upper bound I(P; H)
        mu = self.mu_net(P)
        logvar = self.logvar_net(P)
        positive = -0.5 * ((H - mu) ** 2 / logvar.exp()).sum(dim=1)
        # Negative samples by permutation
        H_shuffle = H[torch.randperm(H.size(0))]
        negative = -0.5 * ((H_shuffle - mu) ** 2 / logvar.exp()).sum(dim=1)
        return (positive - negative).mean()

    def learning_loss(self, P, H):
        mu = self.mu_net(P)
        logvar = self.logvar_net(P)
        return 0.5 * ((mu - H) ** 2 / logvar.exp() + logvar).sum(dim=1).mean()


class GCN_prompt(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, p, cuda_id):
        super(GCN_prompt, self).__init__()
        self.type_p = "GCN"
        if self.type_p == "GAT":
            self.gc1 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, concat=True)
        else:
            self.gc1 = GraphConvolution(nfeat, nhid)

        # self.gc2 = GraphConvolution(nhid, nhid)

        self.dropout = dropout
        self.p = p

        self.device = torch.device(('cuda:' + str(cuda_id)) if torch.cuda.is_available() else 'cpu')

    def forward(self, x, adj, adj_a, att=False):
        x = F.relu(self.gc1(x, adj))
        x_promt1 = x
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        # x_promt2 = x
        if att:
            return x_promt1, x_promt1
        return x_promt1


class Vector_prompt1(nn.Module):
    def __init__(self, nfeat, num_nodes, cuda_id):
        super(Vector_prompt1, self).__init__()
        self.type_p = "GCN"
        self.generator = nn.Parameter(torch.randn(num_nodes, nfeat))
        self.generator.type_p = "GCN"


        self.device = torch.device(('cuda:' + str(cuda_id)) if torch.cuda.is_available() else 'cpu')

    def forward(self, x, adj, adj_a, att=False):
        x_promt1 = self.generator
        return x_promt1


class SimplePrompt(nn.Module):
    def __init__(self, in_channels: int):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()
        self.type_p = "GCN"

    def reset_parameters(self):
        glorot(self.global_emb)

    def forward(self, x, adj, adj_a):
        return self.global_emb

class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()
        self.type_p = "GCN"

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def forward(self, x, adj, adj_a):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)
        return p

class x_prompt(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, p, cuda_id):
        super(x_prompt, self).__init__()
        self.type_p = "GCN"
        self.gc1 = nn.Sequential(
                    nn.Linear(nfeat, nfeat))

        # self.gc2 = GraphConvolution(nhid, nhid)

        self.dropout = dropout
        self.p = p

        self.device = torch.device(('cuda:' + str(cuda_id)) if torch.cuda.is_available() else 'cpu')

    def forward(self, x, adj, adj_a, att=False):
        # x_promt1 = self.gc1(x, adj)
        x = F.relu(self.gc1(x))
        x_promt1 = x
        if att:
            return x_promt1, x_promt1
        return x_promt1
               
class Node_prompt(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, p, cuda_id, IB=True, attack_iters=1, step_size=0.02, gnn='gcn',
                 mask_type="adv", prompt_type='dynamic', norm_if=False, lp=0.2, num_nodes=0):
        super(Node_prompt, self).__init__()
        if gnn == 'gcn':
            prompt_type = prompt_type
            if prompt_type == 'dynamic':
                self.generator = DynamicPrompt(nfeat, nfeat, nhid, dropout, p, cuda_id, norm_if=norm_if, lp=lp)
            elif prompt_type == 'vector':
                assert num_nodes != 0
                self.generator = Vector_prompt1(nfeat, num_nodes, cuda_id)
            elif prompt_type == 'CPF':
                self.generator = SimplePrompt(nfeat)
            elif prompt_type == 'CPFplus':
                self.generator = GPFplusAtt(nfeat, 10)
            elif prompt_type == 'x':
                self.generator = x_prompt(nfeat, nfeat, nhid, dropout, p, cuda_id)
            else:
                self.generator = GCN_prompt(nfeat, nfeat, nhid, dropout, p, cuda_id)
            self.model_teacher = GCN(nfeat, nhid, nclass, dropout, p, cuda_id)
            self.IB = IB
            self.mask_type = mask_type


        for p in self.model_teacher.parameters():
            p.requires_grad = False

        self.eps = 0.1
        self.step_size = step_size
        self.attack_iters = attack_iters
        self.device = torch.device(('cuda:' + str(cuda_id)) if torch.cuda.is_available() else 'cpu')

    def get_prompt(self, x, adj_a, adj_f, att):
        x_promt1, att = self.generator(x, adj_f, adj_a, att)
        return x_promt1, att

    def forward(self, x, adj_a, adj_f, y=None, alpha=None, train=False, dataset_str=None):
        x_raw = x
        softmax_chose = True

        if train:
            mask_type = self.mask_type

            if mask_type == "adv":
                adj_raw = torch.where(adj_a > 0, 1, 0)
                # 1) initialize differentiable Δ
                delta = torch.zeros_like(adj_a, requires_grad=True)


                for _ in range(self.attack_iters):
                    A_pert_raw = torch.clamp(adj_raw + delta, min=0.0,
                                             max=1.0)


                    A_pert_norm = normalize_adj_tensor(
                        A_pert_raw)  # D^{-1/2} A D^{-1/2} :contentReference[oaicite:2]{index=2}


                    x_prompt = self.generator(x, adj_f, A_pert_norm)
                    x_in = 0.5 * (x + x_prompt)


                    out_student = self.model_teacher(x_in, A_pert_norm, softmax_chose=softmax_chose)
                    out_teacher = self.model_teacher(x_raw, adj_a, softmax_chose=softmax_chose)
                    if dataset_str != 'twitch-e':
                        loss_adv = F.kl_div(out_student, out_teacher, reduction='batchmean', log_target=True)
                    if dataset_str == 'twitch-e':
                        loss_adv = torch.nn.BCEWithLogitsLoss()(out_student, y.squeeze(1).to(torch.float))

                    loss_adv.backward()
                    one_hot = False
                    if one_hot:
                        values, indices = torch.topk(delta.grad, k=1, sorted=True)
                        indices = indices[:,0]
                        row = torch.arange(indices.shape[0]).to(indices.device)
                        change = delta.grad[row, indices].sign()
                        delta.grad.zero_()
                        A_adv_raw = adj_raw.clone()
                        A_adv_raw[row, indices] = torch.clamp(A_adv_raw[row, indices] + change, 0.0, 1.0).long()
                    else:
                        grad_sign = delta.grad.sign()
                        delta.data += self.step_size * grad_sign
                        delta.data = torch.clamp(delta.data, -self.eps,
                                                 self.eps)
                        delta.grad.zero_()


                        A_adv_raw = torch.clamp(adj_raw + delta.detach(), 0.0, 1.0)


                    adj_adv = normalize_adj_tensor(A_adv_raw)

                    x_prompt = self.generator(x, adj_f, adj_adv)
                    x_in = 0.5 * (x + x_prompt)
                    out = self.model_teacher(x_in, adj_adv, softmax_chose=softmax_chose)
                    out_teacher = self.model_teacher(x_raw, adj_a, return_x=True, softmax_chose=softmax_chose)
                    if self.IB:
                        H_noisy = self.model_teacher(x_raw, adj_adv)
                        H_noisy = (H_noisy - H_noisy.mean(0)) / (H_noisy.std(0) + 1e-6)
                        # loss_ib = self.club_module(x_prompt, H_noisy)
                        # loss_club_train = self.club_module.learning_loss(x_prompt.detach(), H_noisy.detach())
                        return out, out_teacher, x_prompt, H_noisy
                else:
                    return out, out_teacher

        else:
            x_promt1 = self.generator(x, adj_f, adj_a)
            x = (x + x_promt1) / 2
            out = self.model_teacher(x, adj_a, softmax_chose=softmax_chose)
            return out

    def change_test(self, x, adj_a, adj_f, adj_change, y=None, alpha=None, train=False, dataset_str=None):

        softmax_chose = True

        x_promt1 = self.generator(x, adj_f, adj_change)
        x = (x + x_promt1) / 2
        out = self.model_teacher(x, adj_a, softmax_chose=softmax_chose)
        return out

            
    def get_embd(self, x, adj_a, adj_f, y=None, alpha=None, train=False, dataset_str=None):
        x_raw = x
        x_promt1 = self.generator(x, adj_f, adj_a)
        x = (x + x_promt1) / 2
        out, rep = self.model_teacher(x, adj_a, return_x=True)
        return out, rep
        
class Vector_prompt(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, p, cuda_id, num_nodes, gnn='gcn'):
        super(Vector_prompt, self).__init__()
        self.IB = False
        if gnn == 'gcn':
            self.generator = nn.Parameter(torch.randn(num_nodes, nfeat))
            self.generator.type_p = "GCN"
            self.model_teacher = GCN(nfeat, nhid, nclass, dropout, p, cuda_id)

        for p in self.model_teacher.parameters():
            p.requires_grad = False

    def forward(self, x, adj_a, adj_f, alpha, train=False, dataset_str=None):
        x_raw = x

        if train:
            mask_type = 'load'
            if mask_type == 'random':
                mask_ratio = (torch.randint(1, 6, [1]) / 10).item()
                edge_index = adj_a.nonzero()
                chose_edge_index = torch.randint(0, edge_index.shape[0], [int(edge_index.shape[0] * mask_ratio)])
                chose_edge = edge_index[chose_edge_index]
                adj_a_del = adj_a.clone()
                adj_a_del[chose_edge[:, 0], chose_edge[:, 1]] = 0
            elif mask_type == "load":
                graph_test = ['_add_0.5', '_add_0.75', '_dele_0.5', '_dele_0.2']
                i = graph_test[3]
                adj_add = sp.load_npz(f"../data/{dataset_str}-graph/{dataset_str}{i}.npz")
                adj_add = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj_add))
                adj_a_del = adj_add.to(x.device)

            x_promt1 = self.generator
            x = (x + x_promt1) / 2
            out = self.model_teacher(x, adj_a_del)

            out_teacher = self.model_teacher(x_raw, adj_a, return_x=True)
            return out, out_teacher
        else:
            x_promt1 = self.generator
            x = (x + x_promt1) / 2
            out = self.model_teacher(x, adj_a)
            return out
