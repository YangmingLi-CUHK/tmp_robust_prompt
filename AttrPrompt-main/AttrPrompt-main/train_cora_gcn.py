"""
Phase 1: Train a 2-layer GCN teacher on a selected Cora Metattack graph.

For poisoning evaluation, --train_ptb selects the graph used for supervised
training, validation, and test. Each checkpoint receives a metadata sidecar
recording and fingerprinting that exact graph.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_prompt1 import GCN
from load_cora_metattack import (
    adjacency_fingerprint,
    load_cora_metattack,
    node_split_fingerprint,
)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    return preds.eq(labels).double().sum() / len(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to Meta_Self/raw/ directory')
    parser.add_argument('--save_root', type=str, default='./save_cora/GCN',
                        help='Where to save pretrained models')
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument(
        '--train_ptb', type=float, default=0.0,
        help='Metattack graph used for teacher train/validation/test')
    parser.add_argument(
        '--strict_no_clean_forward', action='store_true',
        help=('For non-zero train_ptb, do not load the clean adjacency tensor '
              'in this teacher-training process'))
    args = parser.parse_args()

    if args.strict_no_clean_forward and np.isclose(args.train_ptb, 0.0):
        raise ValueError(
            "--strict_no_clean_forward requires --train_ptb > 0.")

    os.makedirs(args.save_root, exist_ok=True)

    adj_clean, _adj_f, features, labels, idx_train, idx_val, idx_test, adjs = \
        load_cora_metattack(
            args.data_root,
            k=10,
            ptb_rates=[args.train_ptb],
            load_clean_adj=not args.strict_no_clean_forward,
            build_feature_adj=False,
            split_ptb=args.train_ptb,
        )
    if args.strict_no_clean_forward and adj_clean is not None:
        raise RuntimeError(
            "Protocol violation: clean adjacency tensor was loaded.")

    adj_train_cpu = adjs[args.train_ptb]
    graph_source = f"Meta_Self_Cora_{args.train_ptb}.pt"
    graph_fingerprint = adjacency_fingerprint(adj_train_cpu)
    split_source = f"Meta_Self_Cora_{args.train_ptb}_idx_[train|val|test].npy"
    split_fingerprint = node_split_fingerprint(
        idx_train, idx_val, idx_test)

    print("\n" + "=" * 72)
    print("TEACHER POISONING PROTOCOL")
    print(f"  Train/validation/test graph: {graph_source}")
    print(f"  Graph fingerprint: {graph_fingerprint}")
    print(f"  Node split source: {split_source}")
    print(f"  Node split fingerprint: {split_fingerprint}")
    print(f"  Clean adjacency tensor present: "
          f"{'YES' if adj_clean is not None else 'NO'}")
    print("  Supervised labels: fixed 5-shot training split")
    print("=" * 72 + "\n")

    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    features = features.to(device)
    adj_train = adj_train_cpu.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    n_class = labels.max().item() + 1
    n_feat = features.shape[1]

    all_test = []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        model = GCN(nfeat=n_feat, nhid=args.hidden, nclass=n_class,
                     dropout=args.dropout, p=0.02, cuda_id=args.cuda_id).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val, best_state, patience = -1.0, None, 0
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            loss = F.nll_loss(
                model(features, adj_train, epoch, test=0)[idx_train],
                labels[idx_train],
            )
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out = model(features, adj_train, epoch, test=1)
                acc_val = accuracy(out[idx_val], labels[idx_val])

            if acc_val > best_val:
                best_val = acc_val.item()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if epoch % 50 == 0 or epoch == args.epochs - 1:
                print(f"  [seed {seed:2d}] ep {epoch:04d}  loss={loss.item():.4f}  "
                      f"val={acc_val.item():.4f}  best_val={best_val:.4f}")

            if patience >= args.patience:
                print(f"  [seed {seed:2d}] early stop @ ep {epoch}")
                break

        if best_state is None:
            raise RuntimeError(f"No teacher checkpoint selected for seed {seed}.")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            best_output = model(features, adj_train, test=1)
            best_test = accuracy(
                best_output[idx_test], labels[idx_test]).item()

        save_path = os.path.join(args.save_root, f'model_{seed}.pth')
        torch.save(best_state, save_path)
        metadata = {
            'protocol_version': 1,
            'seed': seed,
            'teacher_train_ptb': args.train_ptb,
            'graph_source': graph_source,
            'adjacency_fingerprint': graph_fingerprint,
            'split_source': split_source,
            'split_fingerprint': split_fingerprint,
            'clean_adjacency_loaded': adj_clean is not None,
            'train_graph': graph_source,
            'validation_graph': graph_source,
            'test_graph': graph_source,
            'train_split_size': int(len(idx_train)),
            'best_validation_accuracy': best_val,
            'test_accuracy_at_best_validation': best_test,
        }
        metadata_path = os.path.join(
            args.save_root, f'model_{seed}.meta.json')
        with open(metadata_path, 'w', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        all_test.append(best_test)
        print(
            f"  [seed {seed:2d}] saved -> {save_path}  "
            f"best_val={best_val:.4f}  best_test={best_test:.4f}")

    mean_test = float(np.mean(all_test) * 100)
    std_test = float(np.std(all_test) * 100)
    print(f"\nPhase 1 Summary ({args.seeds} seeds, M-{args.train_ptb}):")
    print(f"  Test accuracy: {mean_test:.2f}% +/- {std_test:.2f}%")
    for i, acc in enumerate(all_test):
        print(f"    seed {i}: {acc*100:.2f}%")

    summary = {
        'protocol': {
            'teacher_train_ptb': args.train_ptb,
            'graph_source': graph_source,
            'adjacency_fingerprint': graph_fingerprint,
            'split_source': split_source,
            'split_fingerprint': split_fingerprint,
            'clean_adjacency_loaded': adj_clean is not None,
        },
        'metrics': {
            'test_accuracy_mean_pct': mean_test,
            'test_accuracy_std_pct': std_test,
            'test_accuracy_per_seed_pct': [
                float(acc * 100) for acc in all_test
            ],
        },
    }
    with open(os.path.join(args.save_root, 'summary.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
