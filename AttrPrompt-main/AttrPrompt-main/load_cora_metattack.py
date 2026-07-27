"""
Load Cora Metattack data → AttrPrompt-compatible format.
Supports both local Windows and remote Linux server paths.

Output format (matching AttrPrompt's load_data2 convention):
    adj    : dense [N,N] normalized clean adjacency, or None when clean
             adjacency loading is explicitly disabled
    adj_f  : dense [N,N]  KNN feature graph, same normalization
    features : dense [N,F]  row-normalized
    labels : LongTensor [N]
    idx_train/val/test : LongTensor
    perturbed_adjs : dict {ptb_rate: dense [N,N] normalized adj}
"""
import os
import sys
import hashlib
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import kneighbors_graph
from rate_utils import canonical_rate, canonical_rate_tokens, rate_token


def adjacency_fingerprint(adj):
    """Stable SHA-256 of graph support, including node order and self-loops."""
    support = (adj.detach().cpu() > 0).to(torch.uint8).contiguous().numpy()
    payload = f"{support.shape[0]}x{support.shape[1]}:".encode("ascii")
    return hashlib.sha256(payload + support.tobytes()).hexdigest()


def node_split_fingerprint(idx_train, idx_val, idx_test):
    """Stable SHA-256 of the ordered train/validation/test node indices."""
    digest = hashlib.sha256()
    for name, indices in (
            ('train', idx_train), ('val', idx_val), ('test', idx_test)):
        values = indices.detach().cpu().to(torch.int64).contiguous().numpy()
        digest.update(f"{name}:{len(values)}:".encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


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


def load_cora_metattack(data_root, k=10, ptb_rates=None, load_clean_adj=True,
                         build_feature_adj=True, split_ptb=0.0):
    """
    Load Cora + Metattack perturbed graphs.

    Args:
        data_root: path to .../Meta_Self/raw/  (contains .pt, .npz, .npy files)
        k: KNN graph neighbors
        ptb_rates: list of perturbation rates, e.g. [0.0, 0.05, ..., 0.25]
        load_clean_adj: load Meta_Self_Cora_0.00.pt even when 0.00 is not
            requested. Set False for a strict polluted-graph-only run.
        build_feature_adj: build/load the attribute-only KNN adjacency. The
            supervised GCN teacher does not need it.
        split_ptb: rate-specific node split files to load.

    Returns:
        adj, adj_f, features, labels, idx_train, idx_val, idx_test, perturbed_adjs
    """
    if ptb_rates is None:
        ptb_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    ptb_tokens = canonical_rate_tokens(ptb_rates)
    ptb_rates = [canonical_rate(token) for token in ptb_tokens]
    split_ptb = canonical_rate(split_ptb)

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

    # -- Rate-specific node split indices --
    split_prefix = f"Meta_Self_Cora_{rate_token(split_ptb)}"
    idx_train = torch.LongTensor(np.load(
        os.path.join(data_root, f'{split_prefix}_idx_train.npy')))
    idx_val = torch.LongTensor(np.load(
        os.path.join(data_root, f'{split_prefix}_idx_val.npy')))
    idx_test = torch.LongTensor(np.load(
        os.path.join(data_root, f'{split_prefix}_idx_test.npy')))

    # -- KNN feature graph --
    adj_f = None
    if build_feature_adj:
        knn_cache = os.path.join(data_root, f'knn_{k}.pt')
        adj_f = build_knn_graph(features_dense, k=k, cache_path=knn_cache)

    # -- Clean adjacency (optional for strict polluted-graph-only runs) --
    zero_requested = any(np.isclose(ptb, 0.0) for ptb in ptb_rates)
    adj = None
    if load_clean_adj or zero_requested:
        clean_path = os.path.join(
            data_root, f'Meta_Self_Cora_{rate_token(0)}.pt')
        adj_clean_sp = sparse_pt_to_scipy(clean_path)
        adj = dense_norm_adj(adj_clean_sp)

    # -- Perturbed adjs --
    perturbed_adjs = {}
    for ptb in ptb_rates:
        if np.isclose(ptb, 0.0):
            if adj is None:
                raise RuntimeError("Clean adjacency was requested but not loaded.")
            perturbed_adjs[ptb] = adj
        else:
            pt_path = os.path.join(
                data_root, f'Meta_Self_Cora_{rate_token(ptb)}.pt')
            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"Missing: {pt_path}")
            adj_p_sp = sparse_pt_to_scipy(pt_path)
            perturbed_adjs[ptb] = dense_norm_adj(adj_p_sp)

    print(f"  Loaded Cora: {features_tensor.shape[0]} nodes, {features_tensor.shape[1]} feat, "
          f"{labels.max().item()+1} classes")
    print(f"  Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    print(f"  Split source: {split_prefix}_idx_[train|val|test].npy")
    print(f"  Ptb rates: {ptb_tokens}")
    print(f"  Clean adjacency tensor loaded: {'YES' if adj is not None else 'NO'}")
    return adj, adj_f, features_tensor, labels, idx_train, idx_val, idx_test, perturbed_adjs
