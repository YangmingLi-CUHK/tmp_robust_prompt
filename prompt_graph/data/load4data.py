import torch
import os
import numpy as np
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.data import Data
from easydict import EasyDict
from data_attack_fewshot.attackdata_specified import AttackDataset_specified
from data_pyg.data_pyg import get_dataset
from project_paths import attack_data_root, attack_unit_test_data_root, project_path


# =============================================================================
# Cora 统一数据加载函数 (2026-06-16 统一数据源)
# 数据源: data_attack_fewshot/Cora/shot_5/1/Meta_Self/raw/
# =============================================================================

def load4cora_pretrain(dataname='Cora'):
    """预训练数据加载。
    从统一数据源加载清洁 Cora 图 + L1-归一化特征。
    返回 (data, input_dim, out_dim)，供 NodePretrain → ClusterData 使用。
    """
    from project_paths import project_path
    path = project_path('data_attack_fewshot', dataname, 'shot_5', '1')
    dataset = AttackDataset_specified(
        root=path, name='Attack-' + dataname,
        attackmethod='Meta_Self', ptb_rate='0.0',
        transform=NormalizeFeatures()
    )
    data = dataset[0]
    input_dim = dataset.num_features
    out_dim = dataset.num_classes
    return data, input_dim, out_dim


def load4cora_downstream_clean(dataname='Cora', shot_num=5, run_split=1):
    """清洁下游任务数据加载。
    从统一数据源加载清洁 Cora 图 + L1-归一化特征，
    然后从 data_fewshot 加载 few-shot train/val/test 划分索引。
    返回 (data, dataset)。
    """
    from project_paths import project_path
    path = project_path('data_attack_fewshot', dataname, 'shot_{}'.format(shot_num), str(run_split))
    dataset = AttackDataset_specified(
        root=path, name='Attack-' + dataname,
        attackmethod='Meta_Self', ptb_rate='0.0',
        transform=NormalizeFeatures()
    )
    data = dataset[0]

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    index_path = './data_fewshot/{}/shot_{}/{}/index'.format(dataname, str(shot_num), str(run_split))
    if os.path.exists(index_path):
        train_indices = torch.load(index_path + '/train_idx.pt').type(torch.long)
        val_indices   = torch.load(index_path + '/val_idx.pt').type(torch.long)
        test_indices  = torch.load(index_path + '/test_idx.pt').type(torch.long)

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
    else:
        # 如果索引不存在，自动生成
        os.makedirs(index_path, exist_ok=True)
        whole_train_idx = []
        whole_val_idx   = []
        whole_test_idx  = []
        labels = data.y
        for label in data.y.unique():
            label_indices = (data.y == label).nonzero(as_tuple=False).view(-1)
            label_indices = label_indices[torch.randperm(len(label_indices))]
            train_indices = label_indices[:shot_num]
            train_mask[train_indices] = True

            remaining_indices = label_indices[shot_num:]
            split_point = int(len(remaining_indices) * 0.1)
            val_indices = remaining_indices[:split_point]
            test_indices = remaining_indices[split_point:]

            val_mask[val_indices] = True
            test_mask[test_indices] = True

            whole_train_idx.extend(train_indices.numpy())
            whole_val_idx.extend(val_indices.numpy())
            whole_test_idx.extend(test_indices.numpy())

        whole_train_idx = torch.tensor(whole_train_idx)
        whole_val_idx   = torch.tensor(whole_val_idx)
        whole_test_idx  = torch.tensor(whole_test_idx)

        torch.save(whole_train_idx, os.path.join(index_path, 'train_idx.pt'))
        torch.save(labels[whole_train_idx], os.path.join(index_path, 'train_labels.pt'))
        torch.save(whole_val_idx, os.path.join(index_path, 'val_idx.pt'))
        torch.save(labels[whole_val_idx], os.path.join(index_path, 'val_labels.pt'))
        torch.save(whole_test_idx, os.path.join(index_path, 'test_idx.pt'))
        torch.save(labels[whole_test_idx], os.path.join(index_path, 'test_labels.pt'))

    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask

    return data, dataset


# =============================================================================
# 攻击数据加载（下游客场）
# =============================================================================

def load4node_attack_shot_index(data_dir_name, dataname, attack_method, shot_num, run_split, adaptive=None, adaptive_dict=None):
    assert dataname in ['Cora', 'Citeseer', 'PubMed', 'ogbn-arxiv','Cora_ml'], 'Currently, attacks are only supported for the specified datasets.'
    if adaptive:
        path       = attack_unit_test_data_root()
        dataset    = get_dataset(path, 'Unit-' + dataname, adaptive_dict=adaptive_dict)
        print(adaptive_dict)
    else:
        atk_type   = attack_method.split('-')[0]
        atk_ptb    = attack_method.split('-')[1]
        path       = attack_data_root()
        dataset    = get_dataset(path, 'Attack-' + dataname, atk_type, atk_ptb)
        print(f'Attack method : {atk_type} | Attack ptb : {atk_ptb} | Dataset: {dataname}')

    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]

    print()
    print(data)
    print('===========================================================================================================')

    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    class_counts = {}
    for label in data.y:
        label = label.item()
        class_counts[label] = class_counts.get(label, 0) + 1

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    if adaptive:
        index_path = './{}/{}/{}/{}/{}/{}/shot_{}/{}/index'.format(data_dir_name, dataname, adaptive_dict.scenario, str(adaptive_dict.split), adaptive_dict.adaptive_attack_model, str(adaptive_dict.ptb_rate), str(shot_num), str(run_split))
    else:
        index_path = './{}/{}/shot_{}/{}/index'.format(data_dir_name, dataname, str(shot_num), str(run_split))

    if os.path.exists(index_path):
        train_indices  = torch.load(index_path + '/train_idx.pt').type(torch.long)
        train_lbls     = torch.load(index_path + '/train_labels.pt').type(torch.long).squeeze()

        val_indices    = torch.load(index_path + '/val_idx.pt').type(torch.long)
        val_lbs        = torch.load(index_path + '/val_labels.pt').type(torch.long).squeeze()

        test_indices   = torch.load(index_path + '/test_idx.pt').type(torch.long)
        test_lbls      = torch.load(index_path + '/test_labels.pt').type(torch.long).squeeze()

        train_mask[train_indices]  =  True
        val_mask[val_indices]      =  True
        test_mask[test_indices]    =  True

        print(attack_method)
        print(train_indices)
        print(train_lbls)
    else:
        os.makedirs(index_path, exist_ok=True)
        whole_train_idx = []
        whole_val_idx   = []
        whole_test_idx  = []
        labels = data.y

        train_indices = data.train_mask.nonzero().squeeze(-1)
        train_labels  = data.y[train_indices]
        for label in data.y.unique():
            train_indices_specified_label_index =  torch.nonzero(train_labels == label).squeeze(-1)
            if len(train_indices_specified_label_index) != 0:
                train_indices_specified_label = train_indices[train_indices_specified_label_index]
            else:
                print("Indexes with label {} do not exist in the training set, take them from the entire dataset.".format(label))
                train_indices_specified_label = (data.y == label).nonzero(as_tuple=False).view(-1)

            train_indices_specified_label = train_indices_specified_label[torch.randperm(len(train_indices_specified_label))]
            train_indices_specified_label = train_indices_specified_label[:shot_num]
            train_mask[train_indices_specified_label] = True
            whole_train_idx.extend(train_indices_specified_label.numpy())

        remaining_indices = torch.nonzero(train_mask == False).squeeze(-1)
        remaining_indices = remaining_indices[torch.randperm(len(remaining_indices))]
        split_point = int(len(remaining_indices) * 0.1)

        val_indices = remaining_indices[:split_point]
        test_indices = remaining_indices[split_point:]

        val_mask[val_indices] = True
        test_mask[test_indices] = True

        whole_val_idx.extend(val_indices.numpy())
        whole_test_idx.extend(test_indices.numpy())

        whole_train_idx  = torch.tensor(whole_train_idx)
        whole_val_idx    = torch.tensor(whole_val_idx)
        whole_test_idx   = torch.tensor(whole_test_idx)

        whole_train_labels = labels[whole_train_idx]
        whole_val_labels = labels[whole_val_idx]
        whole_test_labels = labels[whole_test_idx]

        torch.save(whole_train_idx, os.path.join(index_path, 'train_idx.pt'))
        torch.save(whole_train_labels, os.path.join(index_path, 'train_labels.pt'))
        torch.save(whole_val_idx, os.path.join(index_path, 'val_idx.pt'))
        torch.save(whole_val_labels, os.path.join(index_path, 'val_labels.pt'))
        torch.save(whole_test_idx, os.path.join(index_path, 'test_idx.pt'))
        torch.save(whole_test_labels, os.path.join(index_path, 'test_labels.pt'))

    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask

    print(f'len train nodes: {sum(data.train_mask)}')
    print(f'len val   nodes: {sum(data.val_mask)}')
    print(f'len test  nodes: {sum(data.test_mask)}')

    return data, dataset


def load4node_attack_specified_shot_index(data_dir_name, dataname, attack_method, shot_num=10, run_split=1):
    assert dataname in ['Cora', 'Citeseer', 'PubMed', 'ogbn-arxiv'], 'Currently, attacks are only supported for the specified datasets.'

    atk_type   = attack_method.split('-')[0]
    atk_ptb    = attack_method.split('-')[1]
    index_path = './{}/{}/shot_{}/{}/index'.format(data_dir_name, dataname, str(shot_num), str(run_split))

    if os.path.exists(index_path):
        path       = project_path(data_dir_name, dataname, 'shot_{}'.format(shot_num), str(run_split))
        dataset    = AttackDataset_specified(root=path, name='Attack-' + dataname, attackmethod=atk_type, ptb_rate=atk_ptb)
        data = dataset[0]
        train_indices        = torch.load(index_path + '/train_idx.pt').type(torch.long)
        attack_train_indices = data.train_mask.nonzero().squeeze().cpu()
        sorted_train_indices = torch.sort(train_indices).values
        sorted_attack_train_indices = torch.sort(attack_train_indices).values
        index_equal = torch.equal(sorted_train_indices, sorted_attack_train_indices)
        if not index_equal:
            raise Exception("The distribution of the attack does not match the specified distribution.")
        else:
            print("Successfully loaded the attack of dataname with few shot {}, split {}".format(str(shot_num), str(run_split)))
            print(f'Attack method : {atk_type} | Attack ptb : {atk_ptb} | Dataset: {dataname}')
            print('======================')
            print(f'Number of graphs: {len(dataset)}')
            print(f'Number of features: {dataset.num_features}')
            print(f'Number of classes: {dataset.num_classes}')
            print()
            print(data)
            print('===========================================================================================================')
            print(f'Number of nodes: {data.num_nodes}')
            print(f'Number of edges: {data.num_edges}')
            print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
            print(f'Has isolated nodes: {data.has_isolated_nodes()}')
            print(f'Has self-loops: {data.has_self_loops()}')
            print(f'Is undirected: {data.is_undirected()}')
        return data, dataset

    else:
        print("Index for the specified shot and run split does not exist. Generating......")
        path_default = attack_data_root()
        dataset      = get_dataset(path_default, 'Attack-' + dataname, atk_type, 0.0)
        data = dataset[0]

        os.makedirs(index_path, exist_ok=True)
        whole_train_idx = []
        whole_val_idx   = []
        whole_test_idx  = []
        labels = data.y
        for label in data.y.unique():
            label_indices = (data.y == label).nonzero(as_tuple=False).view(-1)
            label_indices = label_indices[torch.randperm(len(label_indices))]
            train_indices = label_indices[:shot_num]
            remaining_indices = label_indices[shot_num:]
            split_point = int(len(remaining_indices) * 0.1)
            val_indices = remaining_indices[:split_point]
            test_indices = remaining_indices[split_point:]

            whole_train_idx.extend(train_indices.numpy())
            whole_val_idx.extend(val_indices.numpy())
            whole_test_idx.extend(test_indices.numpy())

        whole_train_idx  = torch.tensor(whole_train_idx)
        whole_val_idx    = torch.tensor(whole_val_idx)
        whole_test_idx   = torch.tensor(whole_test_idx)

        whole_train_labels = labels[whole_train_idx]
        whole_val_labels = labels[whole_val_idx]
        whole_test_labels = labels[whole_test_idx]

        torch.save(whole_train_idx, os.path.join(index_path, 'train_idx.pt'))
        torch.save(whole_train_labels, os.path.join(index_path, 'train_labels.pt'))
        torch.save(whole_val_idx, os.path.join(index_path, 'val_idx.pt'))
        torch.save(whole_val_labels, os.path.join(index_path, 'val_labels.pt'))
        torch.save(whole_test_idx, os.path.join(index_path, 'test_idx.pt'))
        torch.save(whole_test_labels, os.path.join(index_path, 'test_labels.pt'))

        print(f'len train nodes: {len(whole_train_idx)}')
        print(f'len val   nodes: {len(whole_val_idx)}')
        print(f'len test  nodes: {len(whole_test_idx)}')
        print('train indices: {}'.format(whole_train_idx))

        raise Exception("Generated the specified data split, but it still needs to be attacked.")


# =============================================================================
# NodePretrain（GraphCL 入口）
# =============================================================================

def NodePretrain(dataname='Cora', num_parts=200, preprocess_method='None', split_method='Cluster'):
    """预训练数据准备：加载统一数据源的清洁 Cora → METIS 分簇 → 200 个子图。"""
    data, input_dim, _ = load4cora_pretrain(dataname)

    if preprocess_method == 'svd':
        from torch_geometric.transforms import SVDFeatureReduction
        import pickle as pk
        feature_reduce = SVDFeatureReduction(out_channels=100)
        data = feature_reduce(data)
        pk.dump(data, open('./data/{}_feature_reduced.data'.format(dataname), 'bw'))
        data = pk.load(open('./data/{}_feature_reduced.data'.format(dataname), 'br'))

    if split_method == 'Cluster':
        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)
        data = Data(x=x, edge_index=edge_index)
        graph_list = list(ClusterData(data=data, num_parts=num_parts))
    elif split_method == 'Random Walk':
        from torch_cluster import random_walk
        split_ratio = 0.1
        walk_length = 30
        all_random_node_list = torch.randperm(data.num_nodes)
        selected_node_num_for_random_walk = int(split_ratio * data.num_nodes)
        random_node_list = all_random_node_list[:selected_node_num_for_random_walk]
        walk_list = random_walk(data.edge_index[0], data.edge_index[1], random_node_list, walk_length=walk_length)

        graph_list = []
        skip_num = 0
        for walk in walk_list:
            subgraph_nodes = torch.unique(walk)
            if len(subgraph_nodes) < 5:
                skip_num += 1
                continue
            subgraph_data = data.subgraph(subgraph_nodes)
            graph_list.append(subgraph_data)
        print(f"Total {len(graph_list)} random walk subgraphs with nodes more than 5, and there are {skip_num} skipped subgraphs with nodes less than 5.")
    else:
        print('None split method!')
        exit()

    return graph_list, input_dim


# =============================================================================
# ↓↓↓ 以下为 2026-06-16 精简时删除的旧函数（保留为注释，如需恢复取消注释即可）↓↓↓
# =============================================================================

# # 自定义类，将 Data 列表转换为类似 TUDataset 的数据集
# class CustomTUDataset(InMemoryDataset):
#     def __init__(self, data_list, transform=None, pre_transform=None):
#         super().__init__('.', transform, pre_transform)
#         self.data, self.slices = self.collate(data_list)
#     def __getitem__(self, idx):
#         if isinstance(idx, torch.Tensor):
#             idx = idx.tolist()
#         if isinstance(idx, (list, tuple)):
#             return [super(CustomTUDataset, self).get(i) for i in idx]
#         return super(CustomTUDataset, self).get(idx)
#     def __len__(self):
#         return len(self.slices['x']) - 1
#
#
# def graph_sample_and_save(dataset, k, folder, num_classes):
#     num_graphs = len(dataset)
#     num_test = int(0.8 * num_graphs)
#     labels = torch.tensor([graph.y.item() for graph in dataset])
#     all_indices = torch.randperm(num_graphs)
#     test_indices = all_indices[:num_test]
#     torch.save(test_indices, os.path.join(folder, 'test_idx.pt'))
#     test_labels = labels[test_indices]
#     torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))
#     remaining_indices = all_indices[num_test:]
#     train_indices = []
#     for i in range(num_classes):
#         class_indices = [idx for idx in remaining_indices if labels[idx].item() == i]
#         selected_indices = class_indices[:k]
#         train_indices.extend(selected_indices)
#     train_indices = torch.tensor(train_indices)
#     shuffled_indices = torch.randperm(train_indices.size(0))
#     train_indices = train_indices[shuffled_indices]
#     print('idx_train: ', train_indices)
#     torch.save(train_indices, os.path.join(folder, 'train_idx.pt'))
#     train_labels = labels[train_indices]
#     print("train_labels: ", train_labels)
#     torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
#
#
# def load_data4pretrain(dataname='CiteSeer', num_parts=200):
#     import pickle as pk
#     data = pk.load(open('./Dataset/{}/feature_reduced.data'.format(dataname), 'br'))
#     x = data.x.detach()
#     edge_index = data.edge_index
#     edge_index = to_undirected(edge_index)
#     data = Data(x=x, edge_index=edge_index)
#     input_dim = data.x.shape[1]
#     hid_dim = input_dim
#     graph_list = list(ClusterData(data=data, num_parts=num_parts, save_dir='./Dataset/{}/'.format(dataname)))
#     return graph_list, input_dim, hid_dim
#
#
# def load4graph(dataset_name, shot_num=10, num_parts=None, pretrained=False):
#     from torch_geometric.datasets import TUDataset
#     if dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']:
#         dataset = TUDataset(root='data/TUDataset', name=dataset_name, use_node_attr=True)
#         input_dim = dataset.num_features
#         out_dim = dataset.num_classes
#         dataset = dataset.shuffle()
#         graph_list = [data for data in dataset]
#         if dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY']:
#             graph_list = [g for g in graph_list]
#             node_degree_as_features(graph_list)
#             input_dim = graph_list[0].x.size(1)
#         if pretrained:
#             return input_dim, out_dim, graph_list
#         else:
#             return input_dim, out_dim, dataset
#     if dataset_name in ['PubMed', 'CiteSeer', 'Cora']:
#         from torch_geometric.datasets import Planetoid
#         dataset = Planetoid(root='data/Planetoid', name=dataset_name, transform=NormalizeFeatures())
#         data = dataset[0]
#         num_parts = 200
#         x = data.x.detach()
#         edge_index = data.edge_index
#         edge_index = to_undirected(edge_index)
#         data = Data(x=x, edge_index=edge_index)
#         input_dim = dataset.num_features
#         out_dim = dataset.num_classes
#         dataset = list(ClusterData(data=data, num_parts=num_parts))
#         graph_list = dataset
#         return input_dim, out_dim, graph_list
#
#
# def load4node_shot_index(dataname, preprocess_method, shot_num=10, run_split=1):
#     from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, Flickr, WebKB, Actor
#     from ogb.nodeproppred import PygNodePropPredDataset
#     print(dataname)
#     if dataname in ['PubMed', 'Citeseer', 'Cora']:
#         dataset = Planetoid(root='data/Planetoid', name=dataname, transform=NormalizeFeatures())
#         data = dataset[0]
#     elif dataname in ['Computers', 'Photo']:
#         dataset = Amazon(root='data/amazon', name=dataname)
#     elif dataname == 'Reddit':
#         dataset = Reddit(root='data/Reddit')
#     elif dataname == 'WikiCS':
#         dataset = WikiCS(root='data/WikiCS')
#     elif dataname == 'Flickr':
#         dataset = Flickr(root='data/Flickr')
#     elif dataname in ['Wisconsin', 'Texas']:
#         dataset = WebKB(root='data/'+dataname, name=dataname)
#         data = dataset[0]
#     elif dataname == 'Actor':
#         dataset = Actor(root='data/Actor')
#         data = dataset[0]
#     elif dataname == 'ogbn-arxiv':
#         dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data')
#         data = dataset[0]
#     # ... (余下逻辑省略，完整版见 git history)
#
#
# def load4node_demo1(dataname, preprocess_method, shot_num=10):
#     # ... (完整版见 git history)
#     pass
#
#
# def load4link_prediction_single_graph(dataname, num_per_samples=1):
#     # ... (完整版见 git history)
#     pass
#
#
# def node_degree_as_features(data_list):
#     from torch_geometric.utils import degree
#     for data in data_list:
#         deg = degree(data.edge_index[0], dtype=torch.long)
#         deg = deg.view(-1, 1).float()
#         if data.x is None:
#             data.x = deg
#         else:
#             data.x = torch.cat([data.x, deg], dim=1)
#
#
# def load4link_prediction_multi_graph(dataset_name, num_per_samples=1):
#     # ... (完整版见 git history)
#     pass
#
#
# def load4link(dataname):
#     from torch_geometric.datasets import Planetoid, Reddit, WikiCS, Flickr, WebKB
#     from ogb.nodeproppred import PygNodePropPredDataset
#     print(dataname)
#     if dataname in ['PubMed', 'Citeseer', 'Cora']:
#         dataset = Planetoid(root='data/Planetoid', name=dataname)
#     elif dataname == 'Reddit':
#         dataset = Reddit(root='data/Reddit')
#     elif dataname == 'WikiCS':
#         dataset = WikiCS(root='data/WikiCS')
#     elif dataname == 'Flickr':
#         dataset = Flickr(root='data/Flickr')
#     elif dataname in ['Wisconsin', 'Texas']:
#         dataset = WebKB(root='data/'+dataname, name=dataname)
#     elif dataname == 'ogbn-arxiv':
#         dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data')
#     return dataset
#
#
# def load4node_demo2(dataname):
#     # ... (完整版见 git history)
#     pass

# ↑↑↑ 以上为 2026-06-16 精简时删除的旧函数（保留为注释，如需恢复取消注释即可）↑↑↑
# =============================================================================
