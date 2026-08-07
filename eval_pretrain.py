"""
下游线性探测（Linear Probe）评估脚本：对所有预训练 GraphCL 权重进行排序和筛选。

用法:
    python eval_pretrain.py                              # 评估所有 pre_trained_model_raw/*.pth（clean 图）
    python eval_pretrain.py --top_k 5                    # 仅输出 top-5 结果
    python eval_pretrain.py --checkpoint <PATH>          # 评估单个 checkpoint（clean 图）
    python eval_pretrain.py --checkpoint <PATH> --attack_method Meta_Self-0.05  # 攻击图评估

评估方式: 冻结 backbone → 前向获得 node embedding → LogisticRegression 分类 → test accuracy.
"""
import torch
import os
import glob
import argparse
import numpy as np
from torch_geometric.transforms import NormalizeFeatures

from prompt_graph.model import GCN
from prompt_graph.data import load4cora_downstream_clean


def evaluate_checkpoint(model_path, data, device):
    """加载一个预训练 checkpoint，返回 val/test accuracy（线性探测）。"""
    # Parse hid_dim from filename (e.g. Cora.GraphCL.GCN.64_hidden_dim...)
    import re
    hid_dim = 256  # default
    m = re.search(r'(\d+)_hidden_dim', os.path.basename(model_path))
    if m:
        hid_dim = int(m.group(1))
    gnn = GCN(input_dim=data.x.shape[1], hid_dim=hid_dim, num_layer=2).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    gnn.load_state_dict(state_dict)
    gnn.eval()

    with torch.no_grad():
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        emb = gnn(x, edge_index)
        emb = emb.cpu().numpy()

    emb = (emb - emb.mean(0, keepdims=True)) / (emb.std(0, keepdims=True) + 1e-12)

    labels = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(multi_class='multinomial', max_iter=10000, solver='lbfgs')
    clf.fit(emb[train_mask], labels[train_mask])

    val_acc = clf.score(emb[val_mask], labels[val_mask])
    test_acc = clf.score(emb[test_mask], labels[test_mask])

    torch.cuda.empty_cache()
    return val_acc, test_acc


def parse_checkpoint_name(filename):
    """从文件名中解析超参数。"""
    base = os.path.basename(filename).replace('.pth', '')
    parts = base.split('.')

    info = {'filename': os.path.basename(filename), 'path': filename}

    try:
        i = 0
        while i < len(parts):
            part = parts[i]
            if part.startswith('aug1_'):
                info['aug1'] = part.replace('aug1_', '')
            elif part.startswith('aug2_'):
                info['aug2'] = part.replace('aug2_', '')
            elif part.startswith('ratio_'):
                ratio_val = part.replace('ratio_', '')
                if i + 1 < len(parts) and parts[i + 1].replace('.', '').isdigit() and not parts[i + 1].startswith(('aug', 'seed', 'ratio', 'lr')):
                    ratio_val += '.' + parts[i + 1]
                    i += 1
                info['aug_ratio'] = ratio_val
            elif part.startswith('lr_'):
                lr_val = part.replace('lr_', '')
                if i + 1 < len(parts) and parts[i + 1].replace('.', '').isdigit() and not parts[i + 1].startswith(('aug', 'seed', 'ratio')):
                    lr_val += '.' + parts[i + 1]
                    i += 1
                info['lr'] = lr_val
            elif part.startswith('seed_'):
                info['seed'] = part.replace('seed_', '')
            i += 1

        info.setdefault('aug1', '?')
        info.setdefault('aug2', '?')
        info.setdefault('lr', '?')
        info.setdefault('aug_ratio', '?')
        info.setdefault('seed', '?')
    except Exception:
        pass

    return info


def main():
    parser = argparse.ArgumentParser(description='Linear probe evaluation of pretrained GraphCL checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='./pre_trained_model_raw/')
    parser.add_argument('--checkpoint', type=str, default=None, help='Evaluate a single checkpoint instead of all')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=0, help='Show only top K results (0 = all)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--attack_method', type=str, default=None,
                        help='Evaluate on attacked graph (e.g. Meta_Self-0.05). If not set, uses clean graph.')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    print("Loading downstream data...")
    if args.attack_method is not None:
        from prompt_graph.data import load4node_attack_specified_shot_index
        data, dataset = load4node_attack_specified_shot_index(
            'data_attack_fewshot', args.dataset, args.attack_method,
            shot_num=args.shot, run_split=args.split
        )
        print(f"Attack method: {args.attack_method}")
    else:
        data, dataset = load4cora_downstream_clean(args.dataset, shot_num=args.shot, run_split=args.split)
        print("Data: clean (ptb=0.0)")

    data = data.to(device)
    print(f"Data loaded: {data.num_nodes} nodes, {data.num_edges} edges, {dataset.num_classes} classes")
    print(f"Train/Val/Test: {data.train_mask.sum().item()}/{data.val_mask.sum().item()}/{data.test_mask.sum().item()}")

    # 找 checkpoint
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            checkpoint_files = [args.checkpoint]
        else:
            print(f"Checkpoint not found: {args.checkpoint}")
            return
    else:
        checkpoint_files = sorted(glob.glob(os.path.join(args.checkpoint_dir, '*.pth')))

    print(f"\nEvaluating {len(checkpoint_files)} checkpoint(s)...")

    if len(checkpoint_files) == 0:
        print("No checkpoints found! Exiting.")
        return

    # 逐个评估
    results = []
    for i, ckpt in enumerate(checkpoint_files):
        info = parse_checkpoint_name(ckpt)
        print(f"[{i+1}/{len(checkpoint_files)}] {info['filename']}...", end=' ', flush=True)
        try:
            val_acc, test_acc = evaluate_checkpoint(ckpt, data, device)
            info['val_acc'] = val_acc
            info['test_acc'] = test_acc
            results.append(info)
            print(f"Val: {val_acc:.4f} | Test: {test_acc:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")
            info['val_acc'] = float('nan')
            info['test_acc'] = float('nan')
            results.append(info)

    # 排序输出
    results.sort(key=lambda x: x['test_acc'], reverse=True)

    top_k = args.top_k if args.top_k > 0 else len(results)
    print(f"\n{'='*100}")
    print(f"Top-{top_k} checkpoints (sorted by test accuracy)")
    if args.attack_method:
        print(f"Attack: {args.attack_method}")
    print(f"{'='*100}")
    print(f"{'Rank':<5} {'Test Acc':<10} {'Val Acc':<10} {'Aug1':<8} {'Aug2':<8} {'LR':<8} {'Ratio':<8} {'Seed':<6} {'Filename'}")
    print(f"{'-'*110}")

    for rank, r in enumerate(results[:top_k]):
        seed = r.get('seed', '?')
        aug1 = r.get('aug1', '?')
        aug2 = r.get('aug2', '?')
        lr = r.get('lr', '?')
        ratio = r.get('aug_ratio', '?')
        print(f"{rank+1:<5} {r['test_acc']:.4f}     {r['val_acc']:.4f}     {aug1:<8} {aug2:<8} {lr:<8} {ratio:<8} {seed:<6} {r['filename']}")

    # 按超参组聚合（仅在批量评估时）
    if args.checkpoint is None and len(results) > 1:
        print(f"\n{'='*100}")
        print("Aggregated by hyperparameter group (mean ± std across seeds)")
        print(f"{'='*100}")

        from collections import defaultdict
        groups = defaultdict(list)
        for r in results:
            key = (r.get('aug1', '?'), r.get('aug2', '?'), r.get('lr', '?'), r.get('aug_ratio', '?'))
            groups[key].append(r['test_acc'])

        group_results = []
        for (aug1, aug2, lr, aug_ratio), accs in groups.items():
            accs = [a for a in accs if not np.isnan(a)]
            if len(accs) == 0:
                continue
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            group_results.append((mean_acc, std_acc, aug1, aug2, lr, aug_ratio, len(accs)))

        group_results.sort(key=lambda x: x[0], reverse=True)

        print(f"{'Rank':<5} {'Mean Acc':<12} {'Std':<10} {'Aug1':<8} {'Aug2':<8} {'LR':<8} {'Ratio':<8} {'#Seeds'}")
        print(f"{'-'*90}")
        for rank, (mean_acc, std_acc, aug1, aug2, lr, aug_ratio, n_seeds) in enumerate(group_results):
            print(f"{rank+1:<5} {mean_acc:.4f}       {std_acc:.4f}      {aug1:<8} {aug2:<8} {lr:<8} {aug_ratio:<8} {n_seeds}")

    if results:
        best = results[0]
        print(f"\n{'='*100}")
        print(f"BEST CHECKPOINT: {best['filename']}")
        print(f"  Test Acc: {best['test_acc']:.4f}  |  Val Acc: {best['val_acc']:.4f}")
        best_path = best['path']
        print(f"  Command: --pre_train_model_path '{best_path}'")
        print(f"{'='*100}")


if __name__ == '__main__':
    main()
