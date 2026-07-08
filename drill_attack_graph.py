#!/usr/bin/env python3
"""
下钻污染图内部结构：分析 Metattack 攻击边的分布、训练节点受影响程度、per-class 邻域污染
"""
import torch
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path("data_attack_fewshot")


def load_attack_data(ptb_rate: str):
    """Load attacked graph for a given ptb_rate."""
    from data_attack_fewshot.attackdata_specified import AttackDataset_specified
    from torch_geometric.utils import to_undirected

    path = DATA_DIR / "Cora" / "shot_5" / "1"
    dataset = AttackDataset_specified(
        root=str(path), name="Attack-Cora",
        attackmethod="Meta_Self", ptb_rate=ptb_rate
    )
    data = dataset[0]
    return data


def load_clean_data():
    """Load clean graph."""
    from data_attack_fewshot.attackdata_specified import AttackDataset_specified
    path = DATA_DIR / "Cora" / "shot_5" / "1"
    dataset = AttackDataset_specified(
        root=str(path), name="Attack-Cora",
        attackmethod="Meta_Self", ptb_rate="0.0"
    )
    return dataset[0]


def analyze(clean, attacked, ptb_label, mask_name, mask):
    """Drill into one mask's neighborhood corruption."""
    idx = mask.nonzero(as_tuple=False).squeeze().cpu().numpy()
    labels = attacked.y.cpu().numpy()
    n_class = 7

    # Build edge sets as undirected tuples
    def edge_set(data):
        ei = data.edge_index.cpu().numpy()
        edges = set()
        for i in range(ei.shape[1]):
            u, v = int(ei[0, i]), int(ei[1, i])
            if u > v:
                u, v = v, u
            edges.add((u, v))
        return edges

    clean_edges = edge_set(clean)
    attacked_edges = edge_set(attacked)
    added = attacked_edges - clean_edges
    deleted = clean_edges - attacked_edges

    # Per-class analysis of the MASK nodes
    class_stats = defaultdict(lambda: {
        "n_nodes": 0, "n_clean_neighbors": 0, "n_attack_neighbors": 0,
        "attack_neighbor_classes": Counter(),
        "clean_neighbor_classes": Counter(),
        "n_attack_edges_incident": 0,
    })

    edge_index_np = attacked.edge_index.cpu().numpy()
    # Build adjacency for fast lookup
    adj_clean = defaultdict(set)
    adj_attacked = defaultdict(set)
    for u, v in clean_edges:
        adj_clean[u].add(v)
        adj_clean[v].add(u)
    for u, v in attacked_edges:
        adj_attacked[u].add(v)
        adj_attacked[v].add(u)

    for node in idx:
        c = int(labels[node])
        cs = class_stats[c]
        cs["n_nodes"] += 1

        # All neighbors in attacked graph
        attacked_neighbors = adj_attacked.get(node, set())
        clean_only_neighbors = adj_clean.get(node, set())

        attack_neighbors = attacked_neighbors - clean_only_neighbors  # nodes connected via attack edges
        shared_neighbors = attacked_neighbors & clean_only_neighbors  # original clean neighbors

        cs["n_attack_neighbors"] += len(attack_neighbors)
        cs["n_clean_neighbors"] += len(shared_neighbors)
        cs["n_attack_edges_incident"] += len(attack_neighbors)

        for nb in attack_neighbors:
            cs["attack_neighbor_classes"][int(labels[nb])] += 1
        for nb in shared_neighbors:
            cs["clean_neighbor_classes"][int(labels[nb])] += 1

    return class_stats, len(added), len(deleted)


def main():
    print("=" * 70)
    print("Metattack Graph Drill-Down Analysis")
    print("=" * 70)

    # Load clean reference
    print("\nLoading clean graph (ptb=0.0)...")
    clean = load_clean_data()

    labels = clean.y.cpu().numpy()
    n_class = 7
    class_names = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]

    # Load index
    index_path = DATA_DIR / "Cora" / "shot_5" / "1" / "index"
    train_idx_pt = torch.load(str(index_path / "train_idx.pt")).type(torch.long)

    # Build train/val/test masks (same as attackdata_specified)
    def make_mask(idx):
        mask = np.zeros(labels.shape[0], dtype=bool)
        mask[idx] = 1
        return mask

    train_mask = make_mask(train_idx_pt.numpy())

    print(f"\nClean graph: {clean.num_nodes} nodes, {clean.edge_index.shape[1]//2} undirected edges")

    # Per-class distribution
    print(f"\n=== Per-Class Node Distribution ===")
    for c in range(n_class):
        n_total = (labels == c).sum()
        n_train = (labels[train_mask] == c).sum()
        print(f"  {class_names[c]}: {n_total} total, {n_train} train")

    print(f"\n=== Training Nodes (n={train_mask.sum()}) ===")
    train_labels = labels[train_mask]
    for c in range(n_class):
        print(f"  {class_names[c]}: {int((train_labels == c).sum())} nodes")

    # Analyze each ptb level
    for ptb in ["0.05", "0.10", "0.15", "0.20", "0.25"]:
        print(f"\n{'='*70}")
        print(f"PTB = {ptb}")
        print(f"{'='*70}")

        attacked = load_attack_data(ptb)

        # Overall edge stats
        clean_edges_total = clean.edge_index.shape[1] // 2
        attacked_edges_total = attacked.edge_index.shape[1] // 2
        n_added = attacked_edges_total - clean_edges_total  # approximate

        print(f"\nEdge changes: {attacked_edges_total} total (clean={clean_edges_total})")

        for mask_name, mask in [("TRAIN", train_mask)]:
            stats, added_edges, deleted_edges = analyze(clean, attacked, ptb, mask_name, torch.from_numpy(mask))

            print(f"\n  === {mask_name} Mask Per-Class Analysis ===")
            header = f"  {'Class':<8} {'#Nodes':<8} {'CleanNbr':<10} {'AtkNbr':<10} {'TotNbr':<10} {'Atk%':<8} {'Top Attack From':<30}"
            print(header)
            print("  " + "-" * len(header))

            for c in range(n_class):
                cs = stats.get(c, {"n_nodes": 0})
                if cs["n_nodes"] == 0:
                    continue
                n_clean = cs["n_clean_neighbors"]
                n_atk = cs["n_attack_neighbors"]
                total_nbr = n_clean + n_atk
                atk_pct = n_atk / total_nbr * 100 if total_nbr > 0 else 0
                top = cs["attack_neighbor_classes"].most_common(3)
                top_str = ", ".join(f"{class_names[t[0]]}:{t[1]}" for t in top)

                bar = "█" * int(atk_pct / 5) if atk_pct > 0 else ""
                print(f"  {class_names[c]:<8} {cs['n_nodes']:<8} {n_clean:<10} {n_atk:<10} {total_nbr:<10} {atk_pct:<7.1f}% {bar:<8} {top_str:<30}")

        # Global: which classes are attack edges connecting?
        print(f"\n  === Attack Edge Class-Class Connectivity ===")
        def edge_set(data):
            ei = data.edge_index.cpu().numpy()
            edges = set()
            for i in range(ei.shape[1]):
                u, v = int(ei[0, i]), int(ei[1, i])
                if u > v: u, v = v, u
                edges.add((u, v))
            return edges

        clean_e = edge_set(clean)
        attacked_e = edge_set(attacked)
        added_e = attacked_e - clean_e

        class_pair_counts = Counter()
        for u, v in added_e:
            cu, cv = min(labels[u], labels[v]), max(labels[u], labels[v])
            class_pair_counts[(int(cu), int(cv))] += 1

        print(f"  Total added undirected edges: {len(added_e)}")
        print(f"  Top class-class pairs connected by attack edges:")
        for (c1, c2), count in class_pair_counts.most_common(15):
            print(f"    {class_names[c1]} ↔ {class_names[c2]}: {count} edges")

        # Bridge analysis: which class pairs are most targeted?
        print(f"\n  === Attack Bridging: Training Node Classes Connected to Other Classes ===")
        for c in range(n_class):
            # Find training nodes of this class
            train_of_class = set(
                int(node) for node in train_idx_pt.numpy()
                if labels[node] == c
            )
            if not train_of_class:
                continue

            # What classes do attack edges connect these training nodes to?
            atk_to = Counter()
            for node in train_of_class:
                # Find all attack edges incident to this node
                for u, v in added_e:
                    if u == node:
                        atk_to[int(labels[v])] += 1
                    elif v == node:
                        atk_to[int(labels[u])] += 1

            top_targets = atk_to.most_common(5)
            top_str = ", ".join(f"{class_names[t]}:{n}" for t, n in top_targets if t != c)
            print(f"    Train {class_names[c]} (n={len(train_of_class)}): attack edges bridge to → {top_str}")


if __name__ == "__main__":
    main()
