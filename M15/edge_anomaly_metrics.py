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
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _nan_threshold_metrics():
    return {
        "best_f1": math.nan,
        "best_threshold": math.nan,
        "best_precision": math.nan,
        "best_recall": math.nan,
        "best_pred_pos_rate": math.nan,
    }


def _valid_scores(y_score):
    return (
        y_score is not None
        and len(y_score) > 0
        and all(score is not None and math.isfinite(float(score)) for score in y_score)
    )


def _best_threshold_metrics(y_true, y_score):
    if not _valid_scores(y_score):
        return _nan_threshold_metrics()

    y_score = [float(score) for score in y_score]
    thresholds = sorted(set(y_score), reverse=True)
    if not thresholds:
        return _nan_threshold_metrics()

    best = None
    n_edges = len(y_true)
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_score]
        metrics = _binary_metrics(y_true, y_pred)
        pred_pos_rate = sum(y_pred) / n_edges if n_edges > 0 else 0.0
        candidate = {
            "best_f1": metrics["f1"],
            "best_threshold": threshold,
            "best_precision": metrics["precision"],
            "best_recall": metrics["recall"],
            "best_pred_pos_rate": pred_pos_rate,
        }
        if best is None or candidate["best_f1"] > best["best_f1"]:
            best = candidate

    return best if best is not None else _nan_threshold_metrics()


def _precision_recall_at_k_pos(y_true, y_score):
    n_pos = sum(y_true)
    if n_pos <= 0 or not _valid_scores(y_score):
        return {
            "precision_at_k_pos": math.nan,
            "recall_at_k_pos": math.nan,
        }

    y_score = [float(score) for score in y_score]
    ranked_indices = sorted(range(len(y_true)), key=lambda idx: y_score[idx], reverse=True)
    top_k = ranked_indices[:n_pos]
    hits = sum(y_true[idx] for idx in top_k)
    return {
        "precision_at_k_pos": hits / n_pos,
        "recall_at_k_pos": hits / n_pos,
    }


def _auc_metrics(y_true, y_score):
    if y_score is None or len(set(y_true)) < 2:
        return {"auc": math.nan, "ap": math.nan}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return {
            "auc": float(roc_auc_score(y_true, y_score)),
            "ap": float(average_precision_score(y_true, y_score)),
        }
    except Exception:
        return {"auc": math.nan, "ap": math.nan}


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
    anomaly_score = _as_cpu_list(anomaly_score, num_input_edges, default=None)

    by_edge = {}
    for idx, (src, dst) in enumerate(attack_edge_index.t().tolist()):
        if src == dst:
            continue
        key = _canonical_key(src, dst, undirected=undirected)
        record = by_edge.setdefault(
            key,
            {
                "label": 1 if key in added_edges else 0,
                "pred": 0,
                "score": None,
            },
        )
        record["pred"] = max(record["pred"], int(bool(pred_anomaly_mask[idx])))
        score = anomaly_score[idx]
        if score is not None:
            score = float(score)
            if record["score"] is None or score > record["score"]:
                record["score"] = score

    if not by_edge:
        return None

    records = list(by_edge.values())
    y_true = [r["label"] for r in records]
    y_pred = [r["pred"] for r in records]
    explicit_score = [r["score"] for r in records]
    y_score = [r["score"] if r["score"] is not None else float(r["pred"]) for r in records]

    n_edges = len(records)
    n_pos = sum(y_true)
    n_neg = n_edges - n_pos
    pred_pos = sum(y_pred)
    pos_rate = n_pos / n_edges if n_edges > 0 else 0.0
    pred_pos_rate = pred_pos / n_edges if n_edges > 0 else 0.0
    all_positive_precision = pos_rate
    all_positive_recall = 1.0 if n_pos > 0 else 0.0
    all_positive_f1 = (
        2 * all_positive_precision * all_positive_recall
        / (all_positive_precision + all_positive_recall)
        if all_positive_precision + all_positive_recall > 0
        else 0.0
    )

    metrics = _binary_metrics(y_true, y_pred)
    metrics.update(_auc_metrics(y_true, y_score))
    metrics.update(_best_threshold_metrics(y_true, explicit_score))
    metrics.update(_precision_recall_at_k_pos(y_true, explicit_score))
    metrics.update(
        {
            "n_edges": n_edges,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_rate": pos_rate,
            "pred_pos": pred_pos,
            "pred_pos_rate": pred_pos_rate,
            "all_positive_f1": all_positive_f1,
            "current_f1_minus_all_positive_f1": metrics["f1"] - all_positive_f1,
            "hard_mask_all_positive_like": int(n_edges > 0 and pred_pos == n_edges),
            "num_edges": n_edges,
            "num_positive": n_pos,
            "num_predicted": pred_pos,
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
