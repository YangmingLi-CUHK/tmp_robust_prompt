import torch
import os
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.data import Data
from  easydict  import EasyDict
from data_attack_fewshot.attackdata_specified import AttackDataset_specified
from data_pyg.data_pyg import get_dataset
from project_paths import attack_data_root, attack_unit_test_data_root, project_path


# =============================================================================
# 统一数据加载函数 -- Cora 单数据集
# 数据源: data_attack_fewshot/Cora/shot_5/1/Meta_Self/raw/
# =============================================================================

def load4cora_pretrain(dataname='Cora'):
    """预训练数据加载（替代 load4node_demo2）。
    从统一数据源加载清洁 Cora 图 + L1-归一化特征。
    返回 (data, input_dim, out_dim)，其中 data 用于 ClusterData 分簇。
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
    """清洁下游任务数据加载（替代 load4node_shot_index）。
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



def load4node_attack_shot_index(data_dir_name, dataname, attack_method, shot_num, run_split, adaptive=None, adaptive_dict=None):
    assert dataname in ['Cora', 'Citeseer', 'PubMed', 'ogbn-arxiv','Cora_ml'], 'Currently, attacks are only supported for the specified datasets.'
    # 自适应攻击加在数据集
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

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    
    # 根据 shot_num 更新训练掩码
    class_counts = {}  # 统计每个类别的节点数
    for label in data.y:
        label = label.item()
        class_counts[label] = class_counts.get(label, 0) + 1

    # 构建 mask
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    # 这个是随机的取shot方法，这里可以优化一下，因为不同的shot对结果的影响很大，尤其是数据集被攻击的情况下
    # index_path = './{}/{}/{}/index/shot_{}/{}'.format(data_dir_name,dataname, attack_method, str(shot_num), str(run_split))
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
        # quit()
    # 如果不存在文件夹，则创建shot num索引文件夹 并保存train val test的索引
    else:
        os.makedirs(index_path, exist_ok=True)
        # 存放对应shot num的train val test索引
        whole_train_idx = []
        whole_val_idx   = []
        whole_test_idx  = []
        labels = data.y

        # 注意！ seed一样的情况下，不管什么run_split都是一样的！要获得不同的run_split要同时改变seed！
        ############################################################################################################################################
        # 从训练集中取shot
        train_indices = data.train_mask.nonzero().squeeze(-1)
        train_labels  = data.y[train_indices]
        for label in data.y.unique():
            train_indices_specified_label_index =  torch.nonzero(train_labels == label).squeeze(-1)
            if len(train_indices_specified_label_index) != 0:
                train_indices_specified_label = train_indices[train_indices_specified_label_index]
            # 有可能存在train当中不存在标签的情况，但很少, 这种情况从所有的标签中找
            else:
                print("Indexes with label {} do not exist in the training set, take them from the entire dataset.".format(label))
                train_indices_specified_label = (data.y == label).nonzero(as_tuple=False).view(-1)

            train_indices_specified_label = train_indices_specified_label[torch.randperm(len(train_indices_specified_label))]
            train_indices_specified_label = train_indices_specified_label[:shot_num]
            train_mask[train_indices_specified_label] = True
            
            whole_train_idx.extend(train_indices_specified_label.numpy())

        # 得到了训练集的shot 索引，从剩下的索引中按比例划分val和test
        remaining_indices = torch.nonzero(train_mask == False).squeeze(-1)
        remaining_indices = remaining_indices[torch.randperm(len(remaining_indices))]
        split_point = int(len(remaining_indices) * 0.1)

        val_indices = remaining_indices[:split_point]
        test_indices = remaining_indices[split_point:]

        val_mask[val_indices] = True
        test_mask[test_indices] = True

        whole_val_idx.extend(val_indices.numpy())
        whole_test_idx.extend(test_indices.numpy())
        ############################################################################################################################################

        ############################################################################################################################################
        # # 从全部的数据中取shot
        # for label in data.y.unique():
        #         label_indices = (data.y == label).nonzero(as_tuple=False).view(-1)

        #         # if len(label_indices) < 3 * shot_num:
        #         #     raise ValueError(f"类别 {label.item()} 的样本数不足以分配到训练集、测试集和验证集。")

        #         label_indices = label_indices[torch.randperm(len(label_indices))]
        #         train_indices = label_indices[:shot_num]
        #         train_mask[train_indices] = True

        #         remaining_indices = label_indices[shot_num:]
        #         split_point = int(len(remaining_indices) * 0.1)  # 验证集占剩余的10%
                
        #         val_indices = remaining_indices[:split_point]
        #         test_indices = remaining_indices[split_point:]

        #         val_mask[val_indices] = True
        #         test_mask[test_indices] = True

        #         whole_train_idx.extend(train_indices.numpy())
        #         whole_val_idx.extend(val_indices.numpy())
        #         whole_test_idx.extend(test_indices.numpy())
        ############################################################################################################################################
                

        whole_train_idx  = torch.tensor(whole_train_idx)
        whole_val_idx    = torch.tensor(whole_val_idx)
        whole_test_idx   = torch.tensor(whole_test_idx)
        
        # shuffled_train_indices = torch.randperm(whole_train_idx.size(0))
        # whole_train_idx = whole_train_idx[shuffled_train_indices]
        whole_train_labels = labels[whole_train_idx]

        # shuffled_val_indices = torch.randperm(whole_val_idx.size(0))
        # whole_val_idx = whole_val_idx[shuffled_val_indices]
        whole_val_labels = labels[whole_val_idx]

        # shuffled_test_indices = torch.randperm(whole_test_idx.size(0))
        # whole_test_idx = whole_test_idx[shuffled_test_indices]
        whole_test_labels = labels[whole_test_idx]




        # 保存文件
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






def load4node_attack_specified_shot_index(data_dir_name, dataname, attack_method, shot_num= 10, run_split = 1):
    assert dataname in ['Cora', 'Citeseer', 'PubMed', 'ogbn-arxiv'], 'Currently, attacks are only supported for the specified datasets.'

    atk_type   = attack_method.split('-')[0]
    atk_ptb    = attack_method.split('-')[1]
    index_path = './{}/{}/shot_{}/{}/index'.format(data_dir_name, dataname, str(shot_num), str(run_split))
    ##### 看一下train indices 可删
    # train_indices        = torch.load(index_path + '/train_idx.pt').type(torch.long)
    # print(train_indices)
    # quit()
    #####
    # 首先判断在指定的shot和split下是否存在index
    # 如果存在index，就表示能够根据指定的划分加载攻击后的数据
    if os.path.exists(index_path):
        path       = project_path(data_dir_name, dataname, 'shot_{}'.format(shot_num), str(run_split))
        dataset    = AttackDataset_specified(root = path, name = 'Attack-' + dataname,  attackmethod = atk_type, ptb_rate=atk_ptb) # , transform=T.NormalizeFeatures()
        data = dataset[0]
        # 判断一下被攻击数据的划分方式是否和index_path当中存的划分一样，训练集即可
        train_indices        = torch.load(index_path + '/train_idx.pt').type(torch.long)
        attack_train_indices = data.train_mask.nonzero().squeeze().cpu()
        # 对两个tensor进行排序
        sorted_train_indices = torch.sort(train_indices).values
        sorted_attack_train_indices = torch.sort(attack_train_indices).values
        # 判断两个排序后的tensor是否相同
        index_equal = torch.equal(sorted_train_indices, sorted_attack_train_indices)
        if not index_equal:
            raise Exception("The distribution of the attack does not match the specified distribution.")
        else:
            # 这里才是完成了的对指定划分进行了攻击数据的加载
            #############################################
            print("Successfully loaded the attack of dataname with few shot {}, split {}".format(str(shot_num), str(run_split)))
            #############################################
            print(f'Attack method : {atk_type} | Attack ptb : {atk_ptb} | Dataset: {dataname}')
            print('======================')
            print(f'Number of graphs: {len(dataset)}')
            print(f'Number of features: {dataset.num_features}')
            print(f'Number of classes: {dataset.num_classes}')
            print()
            print(data)
            print('===========================================================================================================')
            # Gather some statistics about the graph.
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
        # dataset      = AttackDataset_specified(root = path_default, name = 'Attack-' + dataname,  attackmethod = atk_type, ptb_rate=0.0) # , transform=T.NormalizeFeatures()
        # dataset      = AttackDataset(root = path_default, name = 'Attack-' + dataname, attackmethod = atk_type, ptb_rate=0.0) # , transform=T.NormalizeFeatures()
        dataset      = get_dataset(path_default, 'Attack-' + dataname, atk_type, 0.0)
        
        data = dataset[0]  # Get the first graph object.
        # 表示在指定的shot和split下不存在已经生成的索引，所以要根据默认的数据集自己生成并存放的index当中

        # 创建shot num索引文件夹 并保存train val test的索引

        os.makedirs(index_path, exist_ok=True)
        # 存放对应shot num的train val test索引
        whole_train_idx = []
        whole_val_idx   = []
        whole_test_idx  = []
        labels = data.y
        # 注意！ seed一样的情况下，不管什么run_split都是一样的！要获得不同的run_split要同时改变seed！
        for label in data.y.unique():
                label_indices = (data.y == label).nonzero(as_tuple=False).view(-1)

                # if len(label_indices) < 3 * shot_num:
                #     raise ValueError(f"类别 {label.item()} 的样本数不足以分配到训练集、测试集和验证集。")

                label_indices = label_indices[torch.randperm(len(label_indices))]
                train_indices = label_indices[:shot_num]

                remaining_indices = label_indices[shot_num:]
                split_point = int(len(remaining_indices) * 0.1)  # 验证集占剩余的10%
                
                val_indices = remaining_indices[:split_point]
                test_indices = remaining_indices[split_point:]


                whole_train_idx.extend(train_indices.numpy())
                whole_val_idx.extend(val_indices.numpy())
                whole_test_idx.extend(test_indices.numpy())

        whole_train_idx  = torch.tensor(whole_train_idx)
        whole_val_idx    = torch.tensor(whole_val_idx)
        whole_test_idx   = torch.tensor(whole_test_idx)

        # shuffled_train_indices = torch.randperm(whole_train_idx.size(0))
        # whole_train_idx = whole_train_idx[shuffled_train_indices]
        whole_train_labels = labels[whole_train_idx]

        # shuffled_val_indices = torch.randperm(whole_val_idx.size(0))
        # whole_val_idx = whole_val_idx[shuffled_val_indices]
        whole_val_labels = labels[whole_val_idx]

        # shuffled_test_indices = torch.randperm(whole_test_idx.size(0))
        # whole_test_idx = whole_test_idx[shuffled_test_indices]
        whole_test_labels = labels[whole_test_idx]

        # 保存文件
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






# used in pre_train.py
def NodePretrain(dataname='Cora', num_parts=200, preprocess_method = 'None', split_method='Cluster'):
    data, input_dim, _ = load4cora_pretrain(dataname)
    # if dataname in ['PubMed', 'CiteSeer', 'Cora']:
    #     dataset = Planetoid(root='data/Planetoid', name=dataname)
    # elif dataname in ['Computers', 'Photo']:
    #     dataset = Amazon(root='data/amazon', name=dataname)
    # elif dataname == 'Reddit':
    #     dataset = Reddit(root='data/Reddit')
    # elif dataname == 'WikiCS':
    #     dataset = WikiCS(root='data/WikiCS')
    # elif dataname == 'Flickr':
    #     dataset = Flickr(root='data/Flickr')
    # data = dataset[0]

    ####### feature svd
    if preprocess_method == 'svd':
        feature_reduce = SVDFeatureReduction(out_channels=100)
        data = feature_reduce(data)
        pk.dump(data, open('./data/{}_feature_reduced.data'.format(dataname), 'bw'))
        data = pk.load(open('./data/{}_feature_reduced.data'.format(dataname), 'br'))
    ####### feature svd

    if(split_method=='Cluster'):
        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)
        data = Data(x=x, edge_index=edge_index)
        
        graph_list = list(ClusterData(data=data, num_parts=num_parts))
    elif(split_method=='Random Walk'):
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
            if(len(subgraph_nodes)<5):
                skip_num+=1
                continue
            subgraph_data = data.subgraph(subgraph_nodes)

            graph_list.append(subgraph_data)

        print(f"Total {len(graph_list)} random walk subgraphs with nodes more than 5, and there are {skip_num} skipped subgraphs with nodes less than 5.")

    else:
        print('None split method!')
        exit()
    
    return graph_list, input_dim
