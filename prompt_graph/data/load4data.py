import torch
import os
import hashlib
import numpy as np
import scipy.sparse as sp
from types import SimpleNamespace
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, to_undirected
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
        attackmethod='Meta_Self', ptb_rate='0.00',
        transform=NormalizeFeatures()
    )
    data = dataset[0]
    input_dim = dataset.num_features
    out_dim = dataset.num_classes
    return data, input_dim, out_dim


def load4citeseer_pretrain():
    """Load the paper/Nettack Citeseer LCC and normalize its BoW features."""
    from deeprobust.graph.data import Dataset as DeepRobustDataset
    from scipy.sparse.csgraph import connected_components

    dataset_root = project_path('data', 'deeprobust')
    os.makedirs(dataset_root, exist_ok=True)
    numpy_random_state = np.random.get_state()
    try:
        dataset = DeepRobustDataset(
            root=dataset_root,
            name='citeseer',
            setting='nettack',
            seed=15,
        )
    finally:
        np.random.set_state(numpy_random_state)

    num_components = connected_components(
        dataset.adj,
        directed=False,
        return_labels=False,
    )
    if num_components != 1:
        raise RuntimeError(
            f"DeepRobust/Nettack Citeseer is not connected: {num_components} components."
        )

    adjacency = dataset.adj.tocoo()
    features = dataset.features
    if hasattr(features, 'toarray'):
        features = features.toarray()

    data = Data(
        x=torch.as_tensor(np.asarray(features), dtype=torch.float32),
        edge_index=torch.as_tensor(
            np.vstack((adjacency.row, adjacency.col)),
            dtype=torch.long,
        ),
        y=torch.as_tensor(dataset.labels, dtype=torch.long),
    )
    edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    data.edge_index = edge_index
    data = NormalizeFeatures()(data)

    num_classes = int(torch.unique(data.y).numel())
    class_counts = tuple(torch.bincount(data.y, minlength=num_classes).tolist())

    actual_stats = (
        data.num_nodes,
        data.edge_index.size(1) // 2,
        data.num_features,
        num_classes,
    )
    expected_stats = (2110, 3668, 3703, 6)
    expected_class_counts = (115, 463, 388, 304, 532, 308)
    if actual_stats != expected_stats:
        raise RuntimeError(
            "Unexpected Citeseer LCC statistics: "
            f"nodes={actual_stats[0]}, undirected_edges={actual_stats[1]}, "
            f"features={actual_stats[2]}, classes={actual_stats[3]}; "
            f"expected {expected_stats}."
        )
    if class_counts != expected_class_counts:
        raise RuntimeError(
            f"Unexpected Citeseer class counts: {class_counts}; "
            f"expected {expected_class_counts}."
        )

    feature_row_sums = data.x.sum(dim=1)
    if not torch.allclose(
        feature_row_sums,
        torch.ones_like(feature_row_sums),
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError(
            "Citeseer features are not row-normalized or contain zero-feature nodes."
        )

    data.edge_index, _ = add_remaining_self_loops(
        data.edge_index,
        num_nodes=data.num_nodes,
    )
    expected_runtime_edges = 2 * expected_stats[1] + expected_stats[0]
    if data.edge_index.size(1) != expected_runtime_edges:
        raise RuntimeError(
            "Unexpected Citeseer runtime edge count after adding self-loops: "
            f"{data.edge_index.size(1)}; expected {expected_runtime_edges}."
        )

    print(
        "Pretrain graph: DeepRobust/Nettack Citeseer LCC | "
        f"nodes={actual_stats[0]} | paper_edges={actual_stats[1]} | "
        f"runtime_edges={data.edge_index.size(1)} | "
        f"features={actual_stats[2]} | classes={actual_stats[3]}"
    )
    return data, data.num_features, num_classes


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
        attackmethod='Meta_Self', ptb_rate='0.00',
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


def load4node_attack_specified_raw(data_dir_name, dataname, attack_method, shot_num=10, run_split=1):
    """Load the formal Cora attack graph directly from canonical raw files.

    This bypasses PyG ``*_processed`` caches, which may outlive a regenerated
    attack graph on a long-running server checkout.
    """

    if dataname != 'Cora' or shot_num != 5 or run_split != 1:
        raise ValueError(
            "Strict raw attack loading is dedicated to Cora 5-shot/split-1; "
            f"got dataset={dataname}, shot={shot_num}, split={run_split}."
        )
    if '-' not in attack_method:
        raise ValueError(f"Invalid attack method: {attack_method}")
    atk_type, ptb_rate = attack_method.rsplit('-', 1)
    expected_graphs = {
        '0.00': (0, 0, 5278, 13264, '57b4528c357b3b8ff5ed44ecca47f3de42b84f36c4814961e048c48e67bd65ce'),
        '0.05': (257, 6, 5529, 13766, '660e5ad3b7182007c2ba351e0160c20981bb30345246050b8497b43f111edb80'),
        '0.10': (488, 39, 5727, 14162, '412dd306682d73f6a0adf918fac029ce661c50f1c20935f8fb5bdc92d715f74a'),
        '0.15': (728, 63, 5943, 14594, 'e785143e7127598e14465536337a4a55b6f8a6f5888b0cba083e4e9e3cf54f68'),
        '0.20': (976, 79, 6175, 15058, 'cb71b9db4fc517f15c3f2631dd9dc686589cb82e35b30ba307fb525bf440ff31'),
        '0.25': (1212, 107, 6383, 15474, 'ac4192398d9be424fb15d80df2a0090115f49a5f78fee24fe4870a3b55ed2824'),
    }
    if atk_type != 'Meta_Self' or ptb_rate not in expected_graphs:
        raise ValueError(
            "Strict raw loading requires Meta_Self with a canonical two-decimal "
            f"pollution rate; got {attack_method}."
        )

    root = project_path(data_dir_name, dataname, f'shot_{shot_num}', str(run_split))
    raw_dir = project_path(root, atk_type, 'raw')
    prefix = f'{atk_type}_{dataname}_{ptb_rate}'
    required = [
        project_path(raw_dir, f'{dataname}_features.npz'),
        project_path(raw_dir, f'{dataname}_labels.npy'),
        project_path(raw_dir, f'{prefix}.pt'),
        project_path(raw_dir, f'{prefix}_idx_train.npy'),
        project_path(raw_dir, f'{prefix}_idx_val.npy'),
        project_path(raw_dir, f'{prefix}_idx_test.npy'),
    ]
    canonical_index_root = project_path(
        'data_fewshot', dataname, f'shot_{shot_num}', str(run_split), 'index'
    )
    canonical_index_paths = [
        project_path(canonical_index_root, 'train_idx.pt'),
        project_path(canonical_index_root, 'val_idx.pt'),
        project_path(canonical_index_root, 'test_idx.pt'),
    ]
    missing = [str(path) for path in required + canonical_index_paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing canonical raw attack files: {missing}")

    def file_hash(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    expected_file_hashes = {
        required[0]: 'cba12dbb6b543cf81e601fb29eebc8d2897c35d2455ae6b51658dc95c94228e5',
        required[1]: '1f2fde4fd4b4aca1a4ca053376fb00f5ebeb8fa3e04e8b2a9c0bfd273ca1c83b',
        required[2]: expected_graphs[ptb_rate][4],
        required[3]: '1d2230968368cac607798c04c24d6b634ca2c0e92f3149cc48efb1b0d562dec8',
        required[4]: '00838368d5334cfef5493e7b33c635f57efd307c4dbedd602c178c76683db299',
        required[5]: '37fc182a19c25253f522562d4ecd6a533676928f43530ffe781c67d2c342186f',
    }
    for path, expected_hash in expected_file_hashes.items():
        actual_hash = file_hash(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Canonical raw SHA256 mismatch for {path}: "
                f"expected {expected_hash}, got {actual_hash}."
            )

    features = sp.load_npz(required[0]).toarray().astype(np.float32, copy=False)
    labels = np.load(required[1])
    if features.shape != (2708, 1433) or labels.shape != (2708,):
        raise RuntimeError(
            f"Unexpected Cora raw shapes: features={features.shape}, labels={labels.shape}."
        )

    def load_raw_edges(rate):
        adjacency_path = project_path(raw_dir, f'{atk_type}_{dataname}_{rate}.pt')
        adjacency = torch.load(adjacency_path, map_location='cpu', weights_only=False)
        if not isinstance(adjacency, torch.Tensor):
            raise RuntimeError(f"Attack adjacency is not a tensor: {adjacency_path}")
        if adjacency.layout == torch.strided:
            indices = adjacency.nonzero(as_tuple=False).t().contiguous()
        else:
            sparse_adjacency = adjacency.to_sparse_coo().coalesce()
            nonzero = sparse_adjacency.values() != 0
            indices = sparse_adjacency.indices()[:, nonzero]
        pairs = {
            (min(int(source), int(target)), max(int(source), int(target)))
            for source, target in indices.t().tolist()
            if source != target
        }
        return indices, pairs

    raw_edge_index, attack_edges = load_raw_edges(ptb_rate)
    _, clean_edges = load_raw_edges('0.00')
    added_edges = attack_edges - clean_edges
    deleted_edges = clean_edges - attack_edges
    expected_added, expected_deleted, expected_edges, expected_runtime_edges, graph_hash = expected_graphs[ptb_rate]
    actual_graph = (
        len(added_edges),
        len(deleted_edges),
        len(attack_edges),
        2 * len(attack_edges) + 2708,
    )
    expected_graph = (
        expected_added,
        expected_deleted,
        expected_edges,
        expected_runtime_edges,
    )
    if len(clean_edges) != 5278 or actual_graph != expected_graph:
        raise RuntimeError(
            "Canonical corrected-budget graph check failed: "
            f"attack={attack_method}, clean_edges={len(clean_edges)}, "
            f"actual(add,delete,E,runtimeE)={actual_graph}, expected={expected_graph}."
        )

    raw_edge_index, _ = remove_self_loops(raw_edge_index)
    edge_index, _ = add_remaining_self_loops(raw_edge_index, num_nodes=2708)
    data = Data(
        x=torch.from_numpy(features),
        edge_index=edge_index,
        y=torch.as_tensor(labels, dtype=torch.long),
    )
    if data.num_edges != expected_runtime_edges or not data.is_undirected():
        raise RuntimeError(
            "Strict raw loader produced an unexpected runtime graph: "
            f"edges={data.num_edges}, undirected={data.is_undirected()}."
        )

    split_names = ('train', 'val', 'test')
    masks = {}
    for split_name, raw_index_path, canonical_path in zip(
        split_names, required[3:], canonical_index_paths
    ):
        raw_indices = np.load(raw_index_path).astype(np.int64, copy=False)
        canonical_indices = torch.load(
            canonical_path,
            map_location='cpu',
            weights_only=True,
        ).to(torch.long).cpu().numpy()
        if not np.array_equal(np.sort(raw_indices), np.sort(canonical_indices)):
            raise RuntimeError(
                f"Raw {split_name} indices do not match the canonical 5-shot/split-1 index."
            )
        mask = torch.zeros(2708, dtype=torch.bool)
        mask[torch.from_numpy(raw_indices)] = True
        masks[split_name] = mask

    if tuple(int(masks[name].sum()) for name in split_names) != (35, 265, 2408):
        raise RuntimeError("Unexpected canonical Cora split sizes in strict raw loader.")
    if bool((masks['train'] & masks['val']).any()) or bool(
        (masks['train'] & masks['test']).any()
    ) or bool((masks['val'] & masks['test']).any()):
        raise RuntimeError("Canonical Cora train/val/test masks overlap.")

    data.train_mask = masks['train']
    data.val_mask = masks['val']
    data.test_mask = masks['test']
    dataset = SimpleNamespace(num_classes=7, num_features=1433)
    print(
        "Strict attack raw verified | "
        f"path={required[2]} | attack={attack_method} | "
        f"added={len(added_edges)} | deleted={len(deleted_edges)} | "
        f"clean_edges={len(clean_edges)} | attack_edges={len(attack_edges)} | "
        f"runtime_edges={data.num_edges} | raw_sha256={graph_hash}"
    )
    return data, dataset


# =============================================================================
# NodePretrain（GraphCL 入口）
# =============================================================================

def NodePretrain(
    dataname='Cora',
    num_parts=200,
    preprocess_method='None',
    split_method='Cluster',
    svd_out_dim=100,
):
    """预训练数据准备：加载清洁图，再通过 METIS 划分为子图。"""
    if dataname == 'Citeseer':
        data, input_dim, _ = load4citeseer_pretrain()
    else:
        data, input_dim, _ = load4cora_pretrain(dataname)

    preprocess_method = str(preprocess_method).lower()
    if preprocess_method not in {'none', 'svd'}:
        raise ValueError(
            f"Unsupported preprocess_method {preprocess_method!r}; expected 'none' or 'svd'."
        )

    if preprocess_method == 'svd':
        svd_out_dim = int(svd_out_dim)
        max_svd_dim = min(data.num_nodes, data.num_features)
        if not 1 <= svd_out_dim <= max_svd_dim:
            raise ValueError(
                f"svd_out_dim must be in [1, {max_svd_dim}] for {dataname}, "
                f"got {svd_out_dim}."
            )

        original_input_dim = data.num_features
        cache_path = None
        if dataname == 'Citeseer':
            cache_path = project_path(
                'data',
                'deeprobust',
                f'citeseer_nettack_lcc_l1_svd_{svd_out_dim}.pt',
            )

        if cache_path is not None and os.path.isfile(cache_path):
            reduced_x = torch.load(cache_path, map_location='cpu', weights_only=True)
            cache_action = 'loaded'
        else:
            from torch_geometric.transforms import SVDFeatureReduction

            reduced_data = SVDFeatureReduction(out_channels=svd_out_dim)(data)
            reduced_x = reduced_data.x.detach().cpu()
            cache_action = 'computed'

        expected_shape = (data.num_nodes, svd_out_dim)
        if not isinstance(reduced_x, torch.Tensor) or tuple(reduced_x.shape) != expected_shape:
            raise RuntimeError(
                f"Invalid cached SVD features for {dataname}: expected "
                f"{expected_shape}, got {getattr(reduced_x, 'shape', None)}."
            )
        if not torch.isfinite(reduced_x).all():
            raise RuntimeError(f"SVD features for {dataname} contain non-finite values.")
        if cache_action == 'computed' and cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            temporary_cache_path = f"{cache_path}.tmp_{os.getpid()}.pt"
            torch.save(reduced_x, temporary_cache_path)
            os.replace(temporary_cache_path, cache_path)

        data.x = reduced_x
        input_dim = data.num_features
        print(
            f"SVD features {cache_action}: {dataname} "
            f"{original_input_dim}->{input_dim}"
        )

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
