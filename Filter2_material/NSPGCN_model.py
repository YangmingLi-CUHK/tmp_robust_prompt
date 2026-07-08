from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torch_geometric.utils import to_dense_adj
from utils import normalize_adj_tensor

class MLP(torch.nn.Module):
    def __init__(self,num_i,num_h,num_o):
        super(MLP,self).__init__()
        self.linear1=torch.nn.Linear(num_i,num_h)
        self.linear2=torch.nn.Linear(num_h,num_o)
        self.relu=torch.nn.ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)
    
    def get_embs(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class GraphConv(nn.Module):
    def __init__(self, in_size, out_size, bias=True):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_size, out_size, bias)

    def forward(self, adj_norm, feature):
        h = torch.mm(adj_norm, feature)
        return self.linear(h)

class NSPGCN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, n_node, dropout, device):
        super(NSPGCN, self).__init__()
        self.n_node = n_node
        self.device = device
        if num_layers == 1:
            hidden_size = out_size
        self.num_layers = num_layers
        if dropout > 0.:
            self.feat_drop = nn.Dropout(dropout)
        else:
            self.feat_drop = lambda x: x
        self.linears = nn.ModuleList([nn.Linear(in_size, hidden_size, bias=True)])
        self.gnn_layers = nn.ModuleList([GraphConv(in_size, hidden_size)])
        self.linear_transforms = nn.ModuleList([nn.Linear(in_size, 2, bias=True)])
        self.linear_transforms2 = nn.ModuleList([nn.Linear(in_size, 2, bias=True)])
        
        for i in range(1, num_layers):
            if i == num_layers - 1:
                self.linears.append(nn.Linear(hidden_size, out_size, bias=True))
                self.gnn_layers.append(GraphConv(hidden_size, out_size, bias=True))
                self.linear_transforms.append(nn.Linear(hidden_size, 2, bias=True))
                self.linear_transforms2.append(nn.Linear(hidden_size, 2, bias=True))
                
            else:
                self.linears.append(nn.Linear(hidden_size, hidden_size, bias=True))
                self.gnn_layers.append(GraphConv(hidden_size, hidden_size))
                self.linear_transforms.append(nn.Linear(hidden_size, 2, bias=True))
                self.linear_transforms2.append(nn.Linear(hidden_size, 2, bias=True))
        self.weights_init()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def pairwise_sim(self, mat, kernel):
        if kernel == 'linear':
            sim = (mat @ mat.T)
        elif kernel == 'cos':    
            sim =  (mat @ mat.T) / (mat.norm(dim=1, keepdim=True) @ mat.norm(dim=1, keepdim=True).T)
        sim.fill_diagonal_(0.)
        return sim
    
    def get_kNN_graph(self, x, adj, k, order):
        Ak = torch.matrix_power(adj, order)
        Ak.fill_diagonal_(0.)
        neighbors_embs = torch.mm(Ak, x)
        neighbors_sim = self.pairwise_sim(neighbors_embs, 'cos').cpu().data.numpy()
        for i in range(len(neighbors_sim)):
            indices_argsort = np.argsort(neighbors_sim[i])
            neighbors_sim[i, indices_argsort[: -k]] = 0
        " symmetric kNN graph "
        adj_knn = neighbors_sim + neighbors_sim.T - np.diag(np.diag(neighbors_sim))
        adj_knn[adj_knn != 0] = 1
        return torch.tensor(adj_knn).to(self.device)
    
    def get_kNN_graph_inv(self, x, adj, k, order):
        Ak = torch.matrix_power(adj, order)
        Ak.fill_diagonal_(0.)
        neighbors_embs = torch.mm(Ak, x)
        neighbors_sim = self.pairwise_sim(neighbors_embs, 'cos').cpu().data.numpy()
        for i in range(len(neighbors_sim)):
            indices_argsort = np.argsort(neighbors_sim[i])
            neighbors_sim[i, indices_argsort[: k]] = 0
        " symmetric kNN graph "
        adj_knn = neighbors_sim + neighbors_sim.T - np.diag(np.diag(neighbors_sim))
        adj_knn[adj_knn != 0] = 1
        return torch.tensor(adj_knn).to(self.device)
    
    def forward(self, adj_knn1, adj_knn2, adj_knn_inv1, adj_knn_inv2, feature):
        h = feature
                      
        for i, layer in enumerate(self.gnn_layers):
            if i == self.num_layers - 1:
                s_norm = torch.sigmoid(self.linear_transforms[i](h))
                s2_norm = torch.sigmoid(self.linear_transforms2[i](h))
                adj_new = s_norm[:,0].reshape(-1,1).repeat(1,self.n_node) * adj_knn1 + \
                          s_norm[:,1].reshape(-1,1).repeat(1,self.n_node) * adj_knn2
                adj_inv_new = s2_norm[:,0].reshape(-1,1).repeat(1,self.n_node) * adj_knn_inv1 + \
                              s2_norm[:,1].reshape(-1,1).repeat(1,self.n_node) * adj_knn_inv2
                lap_inv_new = torch.eye(adj_inv_new.shape[0]).to(self.device) - adj_inv_new
                h = layer(adj_new, h) + self.linears[i](h) + layer(lap_inv_new, h)
            else:
                h = self.feat_drop(h)
                s_norm = torch.sigmoid(self.linear_transforms[i](h))
                s2_norm = torch.sigmoid(self.linear_transforms2[i](h))
                adj_new = s_norm[:,0].reshape(-1,1).repeat(1,self.n_node) * adj_knn1 + \
                          s_norm[:,1].reshape(-1,1).repeat(1,self.n_node) * adj_knn2
                adj_inv_new = s2_norm[:,0].reshape(-1,1).repeat(1,self.n_node) * adj_knn_inv1 + \
                              s2_norm[:,1].reshape(-1,1).repeat(1,self.n_node) * adj_knn_inv2
                lap_inv_new = torch.eye(adj_inv_new.shape[0]).to(self.device) - adj_inv_new
                h = layer(adj_new, h) + self.linears[i](h) + layer(lap_inv_new, h)
                h = F.relu(h)
        return F.log_softmax(h, dim=1)
    