# 原版 GPromptShield RobustPrompt_T（保留 paper 原始逻辑，仅加 filter_module=None 兼容）
# 来源: GPromptShield-master/prompt_graph/prompt/RobustPrompt_T.py
# 关键差异 vs 我们的版本:
#   1. out_detect_pt = pass (未实现)
#   2. attention 输出被 F.normalize 覆写 → 假 attention
#   3. τ_tune 基于 prompt 修改后的特征 cosine（单次 GNN forward，在 Tune 后半段做）
#   4. MSE loss 基于全图边（剪枝前）
#   5. 无 last_tau_detection / last_tip_detection / filter_module

import torch
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.data import Data
import numpy as np



class RobustPrompt_T(torch.nn.Module):
    def __init__(self, in_channels: int, muti_defense_pt_dict, p_plus, use_attention, num_heads, cosine_constraint, pt_threshold, temperature, weight_mse, weight_kl, weight_constraint, filter_module=None):
        super(RobustPrompt_T, self).__init__()

        self.in_channels           = in_channels
        self.pt_dict  = muti_defense_pt_dict
        self.pt_keys  = self.pt_dict.keys()

        self.use_attention     = use_attention
        self.cosine_constraint = cosine_constraint   # 是否用cosine计算prompt间距离
        self.pt_threshold      = pt_threshold        # 添加final prompt后的修剪超参数
        self.p_plus            = p_plus

        # Tune过程中的不同loss权重和temperature
        self.temperature       = temperature
        self.weight_mse        = weight_mse
        self.weight_kl         = weight_kl
        self.weight_constraint = weight_constraint

        print('use RobustPrompt Tranductive (ORIGINAL code)')
        print('defense pt dict : ', self.pt_dict)
        print('use_attention : ',self.use_attention)
        print('cosine_constraint : ',self.cosine_constraint)
        print('pt_threshold : ',self.pt_threshold)
        print('temperature : ',self.temperature)
        print('weight_mse',self.weight_mse)
        print('weight_kl : ',self.weight_kl)
        print('weight_constraint : ',self.weight_constraint)


        # GPF版本
        if 'sim_pt' in self.pt_keys:
            if self.p_plus:
                self.sim_pt_list = torch.nn.Parameter(torch.Tensor(20, self.in_channels))
                self.sim_pt_a = torch.nn.Linear(self.in_channels, 20)
            else:
                self.prompt_sim_pt  = torch.nn.Parameter(torch.Tensor(1, self.in_channels))
        if 'degree_pt' in self.pt_keys:
            if self.p_plus:
                self.degree_pt_list = torch.nn.Parameter(torch.Tensor(20, self.in_channels))
                self.degree_pt_a = torch.nn.Linear(self.in_channels, 20)
            else:
                self.prompt_degree_pt   = torch.nn.Parameter(torch.Tensor(1, self.in_channels))
        if 'out_detect_pt' in self.pt_keys:
            if self.p_plus:
                self.out_detect_pt_list = torch.nn.Parameter(torch.Tensor(20, self.in_channels))
                self.out_detect_pt_a = torch.nn.Linear(self.in_channels, 20)
            else:
                self.prompt_out_detect_pt  = torch.nn.Parameter(torch.Tensor(1, self.in_channels))
        if 'other_pt' in self.pt_keys:
            if self.p_plus:
                self.other_pt_list = torch.nn.Parameter(torch.Tensor(20, self.in_channels))
                self.other_pt_a = torch.nn.Linear(self.in_channels, 20)
            else:
                self.prompt_other_pt   = torch.nn.Parameter(torch.Tensor(1, self.in_channels))



        # attention  注意要用batch_first 因为nlp默认的是batch_first=False (L,N,Eq) L为目标序列长度，N为批量大小，Eq为查询嵌入维数embed_dim，batch_first=True时(N,L,Eq)
        # 目前head仅支持1，参数太多会过拟合
        assert num_heads == 1
        self.attention_layer = torch.nn.MultiheadAttention(embed_dim = self.in_channels * num_heads, num_heads = num_heads, dropout = 0.0, batch_first=True)
        self.readout_token   = torch.nn.Parameter(torch.Tensor(1, 1, self.in_channels))
        self.reset_parameters()


    def reset_parameters(self):
        if 'sim_pt' in self.pt_keys:
            if self.p_plus:
                glorot(self.sim_pt_list)
                self.sim_pt_a.reset_parameters()
            else:
                glorot(self.prompt_sim_pt)
        if 'degree_pt' in self.pt_keys:
            if self.p_plus:
                glorot(self.degree_pt_list)
                self.degree_pt_a.reset_parameters()
            else:
                glorot(self.prompt_degree_pt)
        if 'out_detect_pt' in self.pt_keys:
            if self.p_plus:
                glorot(self.out_detect_pt_list)
                self.out_detect_pt_a.reset_parameters()
            else:
                glorot(self.prompt_out_detect_pt)
        if 'other_pt' in self.pt_keys:
            if self.p_plus:
                glorot(self.other_pt_list)
                self.other_pt_a.reset_parameters()
            else:
                glorot(self.prompt_other_pt)
        glorot(self.readout_token)



    def get_muti_prompt(self, node_use_each_pt_whole_graph, device):
        muti_prompt   = []
        overlap_list  = []
        for name, param in self.named_parameters():
            if name.startswith('prompt'):
                key = name.split('_', 1)[1]
                if key not in node_use_each_pt_whole_graph:
                    continue  # skip out_detect_pt (pass) etc.
                muti_prompt.append(param)
                overlap_list.append(node_use_each_pt_whole_graph[key])

        muti_prompt = torch.cat(muti_prompt, dim = 0)
        overlap_matrix = torch.eye(len(overlap_list)).to(device)
        for i in range(len(overlap_list)):
            for j in range(i + 1, len(overlap_list)):
                mask = torch.isin(overlap_list[i], overlap_list[j])
                overlap_nodes = overlap_list[i][mask]
                union_nodes = torch.unique(torch.cat((overlap_list[i],overlap_list[j])))
                overlap_matrix[i][j] = len(overlap_nodes) / len(union_nodes)
                overlap_matrix[j][i] = len(overlap_nodes) / len(union_nodes)
        return muti_prompt, overlap_matrix





    def add_muti_pt(self, graph, device):
        node_use_each_pt_whole_graph = {}

        # 单张图要简单很多 要记录的东西比较少
        g = graph
        # 首先用0.初始化并拼接所有的pt长度
        g_mutiftpt_record = torch.zeros(g.num_nodes, len(self.pt_keys) * self.in_channels)
        g_mutiftpt_record = g_mutiftpt_record.to(device)

        # 记录拼接所有pt的整体向量中每个pt的位置
        pt_range_dict = {}
        range_start = 0
        for pt in self.pt_keys:
            pt_range_dict[pt] = [range_start, range_start + self.in_channels]
            range_start += self.in_channels

        x = g.x
        edge_index = g.edge_index
        # 用于记录当前图中所有defense pt用到的节点
        node_use_pt = torch.tensor([]).to(device)

        if 'sim_pt' in self.pt_keys:
            # 相似度(cos (θ))值范围从-1(不相似)到+1(非常相似) nan代表孤立节点，对结果不造成影响
            x_norm = x / torch.sqrt(torch.sum(x * x,dim=1)).unsqueeze(-1)
            e = torch.sum(x_norm[edge_index[0]] * x_norm[edge_index[1]], dim = 1).unsqueeze(-1)
            row, col = edge_index
            c = torch.zeros(x_norm.shape[0], 1).to(device)
            c = c.scatter_add_(dim=0, index=col.unsqueeze(1), src=e)
            deg = degree(col, x.size(0), dtype=x.dtype).unsqueeze(-1)
            csim = c / deg
            csim = csim.squeeze()
            node_use_sim_pt = torch.nonzero(csim <= self.pt_dict['sim_pt']).squeeze(-1)



            #################################
            if self.p_plus and len(node_use_sim_pt) > 0:
                score = self.sim_pt_a(x[node_use_sim_pt])
                sim_weight = F.softmax(score, dim=1)
                self.prompt_sim_pt = sim_weight.mm(self.sim_pt_list)
            #################################




            # 记录当前图中用到pt的node（任何pt）
            node_use_pt = torch.concat((node_use_pt, node_use_sim_pt))
            # 将prompt放到当前图中用sim pt的指定节点上
            if len(node_use_sim_pt) > 0:
                g_mutiftpt_record[node_use_sim_pt, pt_range_dict['sim_pt'][0] : pt_range_dict['sim_pt'][1]] = self.prompt_sim_pt
            # 记录当前图中用到sim_pt的节点
            node_use_each_pt_whole_graph['sim_pt']  = node_use_sim_pt

        if 'degree_pt' in self.pt_keys:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            node_use_degree_pt = torch.nonzero(deg <= self.pt_dict['degree_pt']).squeeze(-1)


            #################################
            if self.p_plus and len(node_use_degree_pt) > 0:
                score = self.degree_pt_a(x[node_use_degree_pt])
                degree_weight = F.softmax(score, dim=1)
                self.prompt_degree_pt = degree_weight.mm(self.degree_pt_list)
            #################################



            # 记录当前图中用到pt的node（任何pt）
            node_use_pt = torch.concat((node_use_pt, node_use_degree_pt))
            # 将prompt放到当前图中用degree pt的指定节点上
            if len(node_use_degree_pt) > 0:
                g_mutiftpt_record[node_use_degree_pt, pt_range_dict['degree_pt'][0] : pt_range_dict['degree_pt'][1]] = self.prompt_degree_pt
            # 记录当前图中用到degree_pt的节点
            node_use_each_pt_whole_graph['degree_pt'] = node_use_degree_pt

        if 'out_detect_pt' in self.pt_keys:
            pass

        if 'other_pt' in self.pt_keys:
            all_nodes    = torch.arange(0, g.num_nodes).to(device)
            all_pt_nodes = torch.unique(node_use_pt)
            mask = ~torch.isin(all_nodes, all_pt_nodes)
            node_use_no_pt = all_nodes[mask]
            if len(node_use_no_pt) > 0:
                if self.pt_dict['other_pt'] == 'all':
                    node_use_other_pt = node_use_no_pt
                elif self.pt_dict['other_pt'].split('-')[0] == 'random':
                    num_samples     = int(float(self.pt_dict['other_pt'].split('-')[1]) * node_use_no_pt.size(0))
                    random_indices  = torch.randint(0, node_use_no_pt.size(0), (num_samples,))
                    node_use_other_pt  = node_use_no_pt[random_indices]

                #################################
                if self.p_plus:
                    score = self.other_pt_a(x[node_use_other_pt])
                    other_weight = F.softmax(score, dim=1)
                    self.prompt_other_pt = other_weight.mm(self.other_pt_list)
                #################################

            else:
                node_use_other_pt = torch.tensor([], dtype=node_use_no_pt.dtype).to(device)
            if len(node_use_other_pt) > 0:
                g_mutiftpt_record[node_use_other_pt, pt_range_dict['other_pt'][0] : pt_range_dict['other_pt'][1]] = self.prompt_other_pt
            node_use_each_pt_whole_graph['other_pt'] = node_use_other_pt




        ######################################################################################
        # 融合不同pt的方式选择
        g_mutiftpt_record = g_mutiftpt_record.reshape(g.num_nodes, len(self.pt_keys), self.in_channels)
        if self.use_attention:
            g_mutiftpt_record = torch.cat([self.readout_token.expand(g.num_nodes, 1, self.in_channels), g_mutiftpt_record], dim=1)
            padding = torch.zeros(self.in_channels).to(device)
            key_padding_mask = torch.all(g_mutiftpt_record == padding, dim=-1, keepdim=True).squeeze(-1)

            g_mutiftpt_output, g_mutiftpt_attn_weights =  self.attention_layer(g_mutiftpt_record, g_mutiftpt_record, g_mutiftpt_record, key_padding_mask = key_padding_mask)
            g_mutiftpt_output = torch.nn.functional.normalize(g_mutiftpt_record, p=1, dim=2) # 为了数据的稳定！非常关键！
            g_mutiftpt_final_output = g_mutiftpt_output[:,0,:]

            node_num_pt = key_padding_mask.sum(-1)
            node_use_no_pt_indices = torch.nonzero(node_num_pt == len(self.pt_keys)).squeeze(-1)
            g_mutiftpt_final_output[node_use_no_pt_indices] = padding
        else:
            padding = torch.zeros(self.in_channels).to(device)
            wo_padding_mask = torch.all(g_mutiftpt_record != padding, dim=-1, keepdim=True)
            node_prompt_len = wo_padding_mask.sum(1)
            g_mutiftpt_final_output = torch.where(node_prompt_len != 0, g_mutiftpt_record.sum(1) / node_prompt_len, padding)
        ######################################################################################

        g_mutiftpt_final_x = g.x + g_mutiftpt_final_output
        g_mutiftpt         = Data(x=g_mutiftpt_final_x, edge_index=g.edge_index, y=g.y)
        return g_mutiftpt, node_use_each_pt_whole_graph







    def Tune(self, graph, gnn, answering, lossfn, opi, device):

        ######################################################################################
        # 放在后面： 对图的特征进行多提示添加后根据添加prompt的特征修剪图
        g_mutiftpt, node_use_each_pt_whole_graph = self.add_muti_pt(graph, device)
        # Prune edge index
        edge_index = g_mutiftpt.edge_index
        cosine_sim = F.cosine_similarity(g_mutiftpt.x[edge_index[0]], g_mutiftpt.x[edge_index[1]])
        threshold = self.pt_threshold
        keep_edges = cosine_sim >= threshold
        pruned_edge_index = edge_index[:, keep_edges]
        pruned_g_mutiftpt = Data(x=g_mutiftpt.x, edge_index=pruned_edge_index, y=g_mutiftpt.y)
        node_emb = gnn(pruned_g_mutiftpt.x, pruned_g_mutiftpt.edge_index)
        ######################################################################################


        out = answering(node_emb)


        # *********************************************  loss PART  ********************************************* #
        ######################################################################################
        # loss_mse 提示整体以同质性假设为导向
        loss_mse = F.mse_loss(node_emb[g_mutiftpt.edge_index[0]], node_emb[g_mutiftpt.edge_index[1]])
        ######################################################################################
        # loss_pt 针对每一个prompt让筛选节点的平均embedding和未筛选节点的平均embedding相似
        loss_pt = 0.
        for pt in [key for key in self.pt_keys if key not in ('other_pt', 'out_detect_pt')]:
            node_use_each_pt_whole_graph[pt] = node_use_each_pt_whole_graph[pt].long()
            all_batch_nodes    = torch.arange(0, node_emb.shape[0]).to(device)
            mask = ~torch.isin(all_batch_nodes, node_use_each_pt_whole_graph[pt])
            node_use_no_pt    = all_batch_nodes[mask]
            if len(node_use_no_pt) == 0 or len(node_use_each_pt_whole_graph[pt]) == 0:
                continue
            global_pt_mean    = torch.mean(node_emb[node_use_each_pt_whole_graph[pt]], dim = 0)
            global_no_pt_mean = torch.mean(node_emb[node_use_no_pt], dim = 0)
            loss_pt_kl = torch.nn.KLDivLoss()(F.log_softmax(global_pt_mean / self.temperature), F.softmax(global_no_pt_mean / self.temperature))
            loss_pt += loss_pt_kl
        ######################################################################################
        # loss_constraint 针对不同的pt进行约束
        if not self.p_plus:
            muti_prompt, overlap_matrix = self.get_muti_prompt(node_use_each_pt_whole_graph, device)
            if muti_prompt.shape[0] >= 2:
                if self.cosine_constraint:
                    dot_product = torch.matmul(muti_prompt, muti_prompt.T)
                    norms = torch.norm(muti_prompt, dim=1)
                    muti_prompt_matrix = dot_product / (norms[:, None] * norms[None, :])
                    loss_constraint = torch.norm(muti_prompt_matrix - overlap_matrix)
                else:
                    loss_constraint = torch.norm(torch.mm(muti_prompt, muti_prompt.T) - overlap_matrix)
            else:
                loss_constraint = 0.
        else:
            loss_constraint = 0.
        ######################################################################################


        loss = lossfn(out[graph.train_mask], graph.y[graph.train_mask]) + self.weight_mse * loss_mse + self.weight_kl * loss_pt + self.weight_constraint * loss_constraint
        opi.zero_grad()
        loss.backward()
        opi.step()
        return loss
