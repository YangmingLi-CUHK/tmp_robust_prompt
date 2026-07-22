"""
Phase 1: Pretrain 2-layer GCN teacher on Cora clean graph (5-shot split).
Saves to {save_root}/model_{seed}.pth

Usage:
    python train_cora_gcn.py --data_root <path> --save_root <path> [--seeds 10]
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_prompt1 import GCN
from load_cora_metattack import load_cora_metattack


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
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    adj, adj_f, features, labels, idx_train, idx_val, idx_test, _ = \
        load_cora_metattack(args.data_root, k=10, ptb_rates=[0.0])

    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    features = features.to(device)
    adj = adj.to(device)
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

        best_val, best_test, best_state, patience = 0, 0, None, 0
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            loss = F.nll_loss(model(features, adj, epoch, test=0)[idx_train], labels[idx_train])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out = model(features, adj, epoch, test=1)
                acc_val = accuracy(out[idx_val], labels[idx_val])
                acc_test = accuracy(out[idx_test], labels[idx_test])

            if acc_val > best_val:
                best_val = acc_val.item()
                best_test = acc_test.item()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if epoch % 50 == 0 or epoch == args.epochs - 1:
                print(f"  [seed {seed:2d}] ep {epoch:04d}  loss={loss.item():.4f}  "
                      f"val={acc_val.item():.4f}  test={acc_test.item():.4f}  best_val={best_val:.4f}")

            if patience >= args.patience:
                print(f"  [seed {seed:2d}] early stop @ ep {epoch}")
                break

        save_path = os.path.join(args.save_root, f'model_{seed}.pth')
        torch.save(best_state, save_path)
        all_test.append(best_test)
        print(f"  [seed {seed:2d}] saved → {save_path}  best_val={best_val:.4f}  best_test={best_test:.4f}")

    print(f"\nPhase 1 Summary ({args.seeds} seeds):")
    print(f"  Test accuracy: {np.mean(all_test)*100:.2f}% ± {np.std(all_test)*100:.2f}%")
    for i, acc in enumerate(all_test):
        print(f"    seed {i}: {acc*100:.2f}%")


if __name__ == '__main__':
    main()
