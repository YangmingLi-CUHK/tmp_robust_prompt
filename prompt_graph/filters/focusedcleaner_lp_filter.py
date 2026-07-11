import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve


class LinkPrediction(torch.nn.Module):
    """FocusedCleaner-LP link predictor: MLP encoder + inner product."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def encode(self, features):
        out = torch.relu(self.linear1(features))
        return self.linear2(out)

    def forward(self, features, link_idx):
        out = self.encode(features)
        logits = (out[link_idx[:, 0]] * out[link_idx[:, 1]]).sum(dim=1)
        return logits


class FocusedCleanerLPFilter(torch.nn.Module):
    """FocusedCleaner-LP as a project-native edge filter.

    The original FocusedCleaner-LP detects low-probability existing links and
    marks their endpoint nodes as victims. For this project, we expose the link
    probability directly as edge_similarity so it fits the existing filter API.
    """

    def __init__(
        self,
        threshold=0.5,
        hidden_dim=0,
        epochs=50,
        lr=0.1,
        neg_ratio=1.0,
        threshold_mode="gmean",
        max_train_pairs=200000,
        pca_dim=0,
    ):
        super().__init__()
        self.mode = "focusedcleaner_lp"
        self.threshold = threshold
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.neg_ratio = neg_ratio
        self.threshold_mode = threshold_mode
        self.max_train_pairs = max_train_pairs
        self.pca_dim = pca_dim

    def forward(self, graph):
        edge_index = graph.edge_index
        device = graph.x.device
        dtype = graph.x.dtype
        if edge_index.numel() == 0:
            empty = torch.empty(0, device=device, dtype=dtype)
            node_score = torch.zeros(graph.num_nodes, device=device, dtype=dtype)
            return {
                "node_mask": torch.ones(graph.num_nodes, device=device, dtype=torch.bool),
                "edge_mask": torch.empty(0, device=device, dtype=torch.bool),
                "node_score": node_score,
                "edge_score": empty,
                "edge_similarity": empty,
                "filter_threshold": torch.tensor(self.threshold, device=device, dtype=dtype),
            }

        features = self._build_features(graph.x.detach())
        pos_pairs, positive_set = self._unique_undirected_edges(edge_index)
        pos_pairs = self._limit_positive_pairs(pos_pairs)
        neg_pairs = self._sample_negative_pairs(
            graph.num_nodes,
            positive_set,
            int(math.ceil(pos_pairs.size(0) * self.neg_ratio)),
        )

        if pos_pairs.numel() == 0 or neg_pairs.numel() == 0:
            return self._fallback_cosine_filter(graph)

        train_pairs = torch.cat([pos_pairs, neg_pairs], dim=0).to(device)
        train_labels = torch.cat(
            [
                torch.ones(pos_pairs.size(0), dtype=dtype),
                torch.zeros(neg_pairs.size(0), dtype=dtype),
            ],
            dim=0,
        ).to(device)

        input_dim = features.size(1)
        hidden_dim = self.hidden_dim if self.hidden_dim > 0 else input_dim
        predictor = LinkPrediction(input_dim, hidden_dim).to(device)
        optimizer = torch.optim.Adam(predictor.parameters(), lr=self.lr)
        pos_weight = torch.tensor(
            [max(neg_pairs.size(0) / max(pos_pairs.size(0), 1), 1.0)],
            device=device,
            dtype=dtype,
        )

        with torch.enable_grad():
            for _ in range(self.epochs):
                optimizer.zero_grad()
                logits = predictor(features, train_pairs)
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    train_labels,
                    pos_weight=pos_weight,
                )
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            train_probs = torch.sigmoid(predictor(features, train_pairs))
            decision_threshold = self._decision_threshold(train_labels, train_probs)
            pred_pairs = edge_index.t().long()
            edge_similarity = torch.sigmoid(predictor(features, pred_pairs)).to(dtype)
            edge_suspicious_score = 1.0 - edge_similarity
            edge_mask = edge_similarity >= decision_threshold
            node_suspicious_score = self._aggregate_node_scores(
                graph.num_nodes,
                edge_index,
                edge_suspicious_score,
                device,
                dtype,
            )
            node_mask = node_suspicious_score <= (1.0 - decision_threshold)

        return {
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "node_score": node_suspicious_score,
            "edge_score": edge_suspicious_score,
            "edge_similarity": edge_similarity,
            "filter_threshold": torch.tensor(decision_threshold, device=device, dtype=dtype),
        }

    def _build_features(self, x):
        features = x.float()
        pca_dim = min(max(int(self.pca_dim), 0), features.size(0) - 1, features.size(1))
        if pca_dim <= 0:
            return features

        centered = features - features.mean(dim=0, keepdim=True)
        try:
            _, _, v = torch.pca_lowrank(centered, q=pca_dim, center=False)
            pca_features = centered @ v[:, :pca_dim]
        except RuntimeError:
            _, _, vh = torch.linalg.svd(centered.cpu(), full_matrices=False)
            pca_features = (centered.cpu() @ vh[:pca_dim].t()).to(features.device)
        return torch.cat([features, pca_features.to(features.dtype)], dim=1)

    def _limit_positive_pairs(self, pos_pairs):
        if self.max_train_pairs <= 0:
            return pos_pairs
        max_pos = max(int(self.max_train_pairs / (1.0 + max(self.neg_ratio, 1e-12))), 1)
        if pos_pairs.size(0) <= max_pos:
            return pos_pairs
        perm = torch.randperm(pos_pairs.size(0))[:max_pos]
        return pos_pairs[perm]

    def _unique_undirected_edges(self, edge_index):
        pairs = []
        positive_set = set()
        for src, dst in edge_index.detach().cpu().long().t().tolist():
            if src == dst:
                continue
            if src > dst:
                src, dst = dst, src
            key = (src, dst)
            if key in positive_set:
                continue
            positive_set.add(key)
            pairs.append(key)
        if not pairs:
            return torch.empty((0, 2), dtype=torch.long), positive_set
        return torch.tensor(pairs, dtype=torch.long), positive_set

    def _sample_negative_pairs(self, num_nodes, positive_set, num_neg):
        max_possible = num_nodes * (num_nodes - 1) // 2 - len(positive_set)
        target = min(max(num_neg, 0), max_possible)
        if target <= 0:
            return torch.empty((0, 2), dtype=torch.long)

        neg_pairs = []
        neg_set = set()
        tries = 0
        max_tries = target * 20 + 100
        while len(neg_pairs) < target and tries < max_tries:
            remaining = target - len(neg_pairs)
            batch_size = max(remaining * 2, 1024)
            src = torch.randint(0, num_nodes, (batch_size,))
            dst = torch.randint(0, num_nodes, (batch_size,))
            for u, v in zip(src.tolist(), dst.tolist()):
                tries += 1
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                key = (u, v)
                if key in positive_set or key in neg_set:
                    continue
                neg_set.add(key)
                neg_pairs.append(key)
                if len(neg_pairs) >= target:
                    break

        if len(neg_pairs) < target:
            for u in range(num_nodes):
                for v in range(u + 1, num_nodes):
                    key = (u, v)
                    if key in positive_set or key in neg_set:
                        continue
                    neg_set.add(key)
                    neg_pairs.append(key)
                    if len(neg_pairs) >= target:
                        break
                if len(neg_pairs) >= target:
                    break

        return torch.tensor(neg_pairs, dtype=torch.long)

    def _decision_threshold(self, labels, probs):
        if self.threshold_mode == "fixed":
            return float(self.threshold)
        labels_np = labels.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        if len(np.unique(labels_np)) < 2:
            return float(self.threshold)

        fpr, tpr, thresholds = roc_curve(labels_np, probs_np)
        valid = np.isfinite(thresholds)
        if not np.any(valid):
            return float(self.threshold)
        gmeans = np.sqrt(tpr[valid] * (1.0 - fpr[valid]))
        return float(thresholds[valid][np.argmax(gmeans)])

    def _aggregate_node_scores(self, num_nodes, edge_index, edge_scores, device, dtype):
        node_scores = torch.zeros(num_nodes, device=device, dtype=dtype)
        node_counts = torch.zeros(num_nodes, device=device, dtype=dtype)
        src, dst = edge_index
        node_scores.scatter_add_(0, src, edge_scores)
        node_scores.scatter_add_(0, dst, edge_scores)
        ones = torch.ones_like(edge_scores, dtype=dtype)
        node_counts.scatter_add_(0, src, ones)
        node_counts.scatter_add_(0, dst, ones)
        nonzero = node_counts > 0
        node_scores[nonzero] = node_scores[nonzero] / node_counts[nonzero]
        return node_scores

    def _fallback_cosine_filter(self, graph):
        edge_similarity = F.cosine_similarity(
            graph.x[graph.edge_index[0]],
            graph.x[graph.edge_index[1]],
            dim=1,
            eps=1e-12,
        )
        edge_similarity = (edge_similarity + 1.0) / 2.0
        edge_suspicious_score = 1.0 - edge_similarity
        edge_mask = edge_similarity >= self.threshold
        node_suspicious_score = self._aggregate_node_scores(
            graph.num_nodes,
            graph.edge_index,
            edge_suspicious_score,
            graph.x.device,
            graph.x.dtype,
        )
        node_mask = node_suspicious_score <= (1.0 - self.threshold)
        return {
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "node_score": node_suspicious_score,
            "edge_score": edge_suspicious_score,
            "edge_similarity": edge_similarity,
            "filter_threshold": torch.tensor(self.threshold, device=graph.x.device, dtype=graph.x.dtype),
        }
