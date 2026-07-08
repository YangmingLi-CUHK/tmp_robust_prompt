"""NSP (Neighbor-Similarity-Preserving) anomaly detection for RobustPrompt.

Ported from ``Filter2_material/NSPGCN_model.py`` (paper: "Towards Robust GSL Beyond
Homophily via Preserving Neighbor Similarity"). Reuse authorized by the first author.

Core idea: adversarial edges (e.g. Metattack) tend to connect nodes whose *neighbor
distributions* are dissimilar, even when their raw bag-of-words features look benign.
We therefore measure similarity on higher-order neighbor embeddings

    N = (A^order) X      (self term removed, i.e. diag(A^order) zeroed)

and flag an observed edge (u, v) as anomalous when the cosine similarity of its two
endpoints' neighbor embeddings is low. Because RobustPrompt applies defense prompts at
the *node* level, a flagged edge marks BOTH of its endpoints as anomalous nodes (same
convention as ``out_detect_pt``).

Scale note: the faithful port forms a dense NxN adjacency and uses ``matrix_power``.
This is fine for Cora-scale graphs (our current setting). For much larger graphs the
per-edge neighbor-similarity path below avoids the dense NxN cosine matrix; only the
optional kNN mode materializes it.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .neighbor_similarity_filter import _BaseEdgeFilter


def pairwise_sim(mat, kernel="cos"):
    """Ported from NSPGCN_model.py: dense NxN similarity with the diagonal zeroed."""
    if kernel == "linear":
        sim = mat @ mat.T
    elif kernel == "cos":
        denom = mat.norm(dim=1, keepdim=True) @ mat.norm(dim=1, keepdim=True).T
        sim = (mat @ mat.T) / (denom + 1e-12)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")
    sim.fill_diagonal_(0.0)
    return sim


def dense_adj(edge_index, num_nodes, device):
    """Symmetric 0/1 dense adjacency (no self-loops) from an edge_index."""
    adj = torch.zeros((num_nodes, num_nodes), device=device)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj[edge_index[1], edge_index[0]] = 1.0
    return adj


def neighbor_embeddings(x, adj, order):
    """N = (A^order) X with the self term removed, matching NSPGCN's
    ``Ak = matrix_power(adj, order); Ak.fill_diagonal_(0.)``."""
    Ak = torch.matrix_power(adj, order).clone()
    Ak.fill_diagonal_(0.0)
    return torch.mm(Ak, x)


def get_kNN_graph(nbr_emb, k, inverse=False, device=None):
    """Port of NSPGCN get_kNN_graph / get_kNN_graph_inv (symmetric binary kNN).

    inverse=False keeps each row's k most-similar neighbors (the "similar" graph);
    inverse=True drops each row's k most-dissimilar neighbors (the "inverse" graph).
    """
    sim = pairwise_sim(nbr_emb, "cos").cpu().data.numpy()
    for i in range(len(sim)):
        order = np.argsort(sim[i])
        if inverse:
            sim[i, order[:k]] = 0
        else:
            sim[i, order[:-k]] = 0
    adj_knn = sim + sim.T - np.diag(np.diag(sim))
    adj_knn[adj_knn != 0] = 1
    t = torch.tensor(adj_knn, dtype=torch.float32)
    return t.to(device) if device is not None else t


def nsp_edge_scores(x, edge_index, num_nodes, order=2):
    """Per observed edge: neighbor-embedding cosine similarity and suspicious score.

    Returns ``(edge_similarity, edge_suspicious_score = 1 - edge_similarity)``.
    """
    if edge_index.numel() == 0:
        empty = torch.empty(0, device=x.device, dtype=x.dtype)
        return empty, empty
    adj = dense_adj(edge_index, num_nodes, x.device)
    nbr = neighbor_embeddings(x, adj, order)
    sim = F.cosine_similarity(nbr[edge_index[0]], nbr[edge_index[1]], dim=1, eps=1e-12)
    return sim, 1.0 - sim


def nsp_suspicious_nodes(x, edge_index, num_nodes, sim_threshold, order=2):
    """Flag low-neighbor-similarity edges and return the unique set of endpoint nodes.

    Returns ``(nodes, edge_mask, edge_similarity)`` where ``nodes`` are the anomalous
    node indices (both endpoints of each flagged edge). ``edge_mask``/``edge_similarity``
    are kept so callers can log edge-level anomaly-detection metrics.
    """
    device = x.device
    sim, _ = nsp_edge_scores(x, edge_index, num_nodes, order)
    if sim.numel() == 0:
        empty_nodes = torch.empty(0, dtype=torch.long, device=device)
        return empty_nodes, None, sim
    edge_mask = sim <= sim_threshold
    flagged = edge_index[:, edge_mask]
    if flagged.numel() > 0:
        nodes = torch.unique(torch.cat([flagged[0], flagged[1]]))
    else:
        nodes = torch.empty(0, dtype=torch.long, device=device)
    return nodes, edge_mask, sim


class NSPFilter(_BaseEdgeFilter):
    """Neighbor-Similarity-Preserving filter, compatible with the ``filter_module``
    interface (``forward(graph) -> {node_mask, edge_mask, node_score, edge_score,
    edge_similarity}``) so it can also be plugged in for edge-anomaly-detection metrics.

    Unlike ``OriginalFilter`` (raw-feature cosine) it scores edges by the cosine
    similarity of ``order``-hop neighbor embeddings, capturing structural anomalies that
    raw-feature similarity misses.
    """

    def __init__(self, threshold, order=2):
        super().__init__(threshold=threshold)
        self.mode = "nsp"
        self.order = order

    def compute_edge_statistics(self, graph):
        return nsp_edge_scores(graph.x, graph.edge_index, graph.num_nodes, self.order)
