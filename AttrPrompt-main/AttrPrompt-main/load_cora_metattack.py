"""
Load Cora Metattack data → AttrPrompt-compatible format.
Supports both local Windows and remote Linux server paths.

Output format (matching AttrPrompt's load_data2 convention):
    adj    : dense [N,N]  D^{-1/2}(A+I)D^{-1/2}  normalized clean adjacency
    adj_f  : dense [N,N]  KNN feature graph, same normalization
    features : dense [N,F]  row-normalized
    labels : LongTensor [N]
    idx_train/val/test : LongTensor
    perturbed_adjs : dict {ptb_rate: dense [N,N] normalized adj}
"""
import os
import sys
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import kneighbors_graph


def normalize_adj(adj):
    """Symmetrically normalize adjacency: D^{-1/2} A D^{-1/2} (coo → coo)."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_pt_to_scipy(pt_path):
    """Load a torch.sparse .pt file → scipy csr_matrix (no self-loops)."""
    pt = torch.load(pt_path, map_location='cpu', weights_only=False)
    idx = pt.coalesce().indices()
    vals = pt._values().numpy()
    N = pt.shape[0]
    mask = (idx[0] != idx[1])  # remove self-loops
    row = idx[0][mask].numpy()
    col = idx[1][mask].numpy()
    data = vals[mask.numpy()]
    return sp.csr_matrix((data, (row, col)), shape=(N, N))


def dense_norm_adj(adj_scipy):
    """scipy csr (no self-loops) → dense D^{-1/2}(A+I)D^{-1/2}."""
    N = adj_scipy.shape[0]
    adj_norm = normalize_adj(adj_scipy + sp.eye(N))
    return torch.FloatTensor(np.array(adj_norm.todense()))


def build_knn_graph(features_dense, k=10, cache_path=None):
    """Build k-NN graph (cosine), symmetric + normalize → dense [N,N]."""
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, map_location='cpu', weights_only=True)

    adj_f = kneighbors_graph(features_dense, k, mode='connectivity',
                             metric='cosine', include_self=False)
    adj_f = adj_f + adj_f.T
    adj_f.data = np.ones_like(adj_f.data)
    adj_f_norm = normalize_adj(adj_f + sp.eye(adj_f.shape[0]))
    result = torch.FloatTensor(np.array(adj_f_norm.todense()))

    if cache_path:
        torch.save(result, cache_path)
        print(f"  KNN graph saved to {cache_path}")
    return result


def load_cora_metattack(data_root, k=10, ptb_rates=None):
    """
    Load Cora + Metattack perturbed graphs.

    Args:
        data_root: path to .../Meta_Self/raw/  (contains .pt, .npz, .npy files)
        k: KNN graph neighbors
        ptb_rates: list of perturbation rates, e.g. [0.0, 0.05, ..., 0.25]

    Returns:
        adj, adj_f, features, labels, idx_train, idx_val, idx_test, perturbed_adjs
    """
    if ptb_rates is None:
        ptb_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

    # -- Features (scipy sparse → dense row-normalized) --
    features_sp = sp.load_npz(os.path.join(data_root, 'Cora_features.npz'))
    features_dense = np.array(features_sp.todense())
    rowsum = np.array(features_sp.sum(1)) + 1e-8
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    features_norm = sp.diags(r_inv).dot(features_sp)
    features_tensor = torch.FloatTensor(np.array(features_norm.todense()))

    # -- Labels --
    labels_np = np.load(os.path.join(data_root, 'Cora_labels.npy'))
    labels = torch.tensor(labels_np, dtype=torch.long)

    # -- Split indices (use ptb=0.0 split for all rates; verified identical across ptb) --
    idx_train = torch.LongTensor(np.load(os.path.join(data_root, 'Meta_Self_Cora_0.0_idx_train.npy')))
    idx_val   = torch.LongTensor(np.load(os.path.join(data_root, 'Meta_Self_Cora_0.0_idx_val.npy')))
    idx_test  = torch.LongTensor(np.load(os.path.join(data_root, 'Meta_Self_Cora_0.0_idx_test.npy')))

    # -- KNN feature graph --
    knn_cache = os.path.join(data_root, f'knn_{k}.pt')
    adj_f = build_knn_graph(features_dense, k=k, cache_path=knn_cache)

    # -- Clean adjacency --
    adj_clean_sp = sparse_pt_to_scipy(os.path.join(data_root, 'Meta_Self_Cora_0.0.pt'))
    adj = dense_norm_adj(adj_clean_sp)

    # -- Perturbed adjs --
    perturbed_adjs = {}
    for ptb in ptb_rates:
        if ptb == 0.0:
            perturbed_adjs[ptb] = adj
        else:
            pt_path = os.path.join(data_root, f'Meta_Self_Cora_{ptb}.pt')
            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"Missing: {pt_path}")
            adj_p_sp = sparse_pt_to_scipy(pt_path)
            perturbed_adjs[ptb] = dense_norm_adj(adj_p_sp)

    print(f"  Loaded Cora: {features_tensor.shape[0]} nodes, {features_tensor.shape[1]} feat, "
          f"{labels.max().item()+1} classes")
    print(f"  Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    print(f"  Ptb rates: {list(perturbed_adjs.keys())}")
    return adj, adj_f, features_tensor, labels, idx_train, idx_val, idx_test, perturbed_adjs
