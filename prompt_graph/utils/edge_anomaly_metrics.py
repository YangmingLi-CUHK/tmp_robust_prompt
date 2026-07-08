import math

import torch


def _to_cpu_long(edge_index):
    if edge_index is None:
        return torch.empty((2, 0), dtype=torch.long)
    return edge_index.detach().cpu().long()


def edge_keys(edge_index, undirected=True, remove_self_loops=True):
    edge_index = _to_cpu_long(edge_index)
    keys = []
    if edge_index.numel() == 0:
        return keys

    for src, dst in edge_index.t().tolist():
        if remove_self_loops and src == dst:
            continue
        if undirected and src > dst:
            src, dst = dst, src
        keys.append((src, dst))
    return keys


def _canonical_key(src, dst, undirected=True):
    if undirected and src > dst:
        src, dst = dst, src
    return src, dst


def edge_key_set(edge_index, undirected=True, remove_self_loops=True):
    return set(edge_keys(edge_index, undirected=undirected, remove_self_loops=remove_self_loops))


def pollution_diff(clean_edge_index, attack_edge_index, undirected=True):
    clean_edges = edge_key_set(clean_edge_index, undirected=undirected)
    attack_edges = edge_key_set(attack_edge_index, undirected=undirected)
    return {
        "clean_edges": clean_edges,
        "attack_edges": attack_edges,
        "added_edges": attack_edges - clean_edges,
        "deleted_edges": clean_edges - attack_edges,
    }


def _binary_metrics(y_true, y_pred):
    """纯二元分类指标。TPR=攻击边检出率, TNR=干净边保留率。"""
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    tpr       = recall                          # 攻击边检出率
    tnr       = tn / (tn + fp) if tn + fp > 0 else 0.0  # 干净边保留率
    f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    balanced_accuracy = (tpr + tnr) / 2.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": tpr,
        "tnr": tnr,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def evaluate_edge_detection(
    attack_edge_index,
    added_edges,
    pred_anomaly_mask=None,
    anomaly_score=None,
    undirected=True,
):
    attack_edge_index = _to_cpu_long(attack_edge_index)
    num_input_edges = attack_edge_index.size(1)
    if num_input_edges == 0:
        return None

    pred_anomaly_mask = _as_cpu_list(pred_anomaly_mask, num_input_edges, default=False)

    by_edge = {}
    for idx, (src, dst) in enumerate(attack_edge_index.t().tolist()):
        if src == dst:
            continue
        key = _canonical_key(src, dst, undirected=undirected)
        record = by_edge.setdefault(
            key,
            {"label": 1 if key in added_edges else 0, "pred": 0},
        )
        record["pred"] = max(record["pred"], int(bool(pred_anomaly_mask[idx])))

    if not by_edge:
        return None

    records = list(by_edge.values())
    y_true = [r["label"] for r in records]
    y_pred = [r["pred"] for r in records]

    metrics = _binary_metrics(y_true, y_pred)
    metrics.update(
        {
            "num_edges": len(records),
            "num_positive": sum(y_true),
            "num_predicted": sum(y_pred),
        }
    )
    return metrics


def evaluate_node_detection(num_nodes, added_edges, pred_nodes):
    polluted_nodes = set()
    for src, dst in added_edges:
        polluted_nodes.add(src)
        polluted_nodes.add(dst)

    pred_nodes = set(_as_node_list(pred_nodes))
    y_true = [1 if node in polluted_nodes else 0 for node in range(num_nodes)]
    y_pred = [1 if node in pred_nodes else 0 for node in range(num_nodes)]

    metrics = _binary_metrics(y_true, y_pred)
    metrics.update(
        {
            "num_nodes": num_nodes,
            "num_positive": sum(y_true),
            "num_predicted": sum(y_pred),
        }
    )
    return metrics


def evaluate_incident_edge_detection(attack_edge_index, added_edges, pred_nodes, undirected=True):
    pred_nodes = set(_as_node_list(pred_nodes))
    attack_edge_index = _to_cpu_long(attack_edge_index)
    pred_mask = [
        (src in pred_nodes or dst in pred_nodes)
        for src, dst in attack_edge_index.t().tolist()
    ]
    return evaluate_edge_detection(
        attack_edge_index,
        added_edges,
        pred_anomaly_mask=pred_mask,
        anomaly_score=None,
        undirected=undirected,
    )


def _as_cpu_list(values, length, default=None):
    if values is None:
        return [default] * length
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().view(-1).tolist()
    else:
        values = list(values)
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {len(values)}")
    return values


def _as_node_list(nodes):
    if nodes is None:
        return []
    if isinstance(nodes, torch.Tensor):
        if nodes.numel() == 0:
            return []
        return nodes.detach().cpu().long().view(-1).tolist()
    out = []
    for node in nodes:
        if isinstance(node, (list, tuple, set)):
            out.extend(_as_node_list(node))
        else:
            out.append(int(node))
    return out


def fmt_metric(value):
    if value is None:
        return "nan"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{float(value):.4f}"
