"""
Phase 2: Train AttrPrompt Node_prompt on 5-shot Cora + Metattack perturbed graphs.

Key: KL-divergence distillation — student (prompt + adversarially perturbed adj)
must match frozen teacher (clean adj, no prompt) predictions on all nodes.

Cora config matches the paper: norm_if=True, IB=True by default.

Usage:
    python train_cora_attrprompt.py --data_root <path> --pretrain_root <path> [--seeds 10]
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_prompt1 import Node_prompt
from load_cora_metattack import load_cora_metattack


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    return preds.eq(labels).double().sum() / len(labels)


def f1_macro(output, labels):
    from sklearn.metrics import f1_score
    return f1_score(output.max(1)[1].cpu(), labels.cpu(), average='macro')


def main():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to Meta_Self/raw/ directory')
    parser.add_argument('--pretrain_root', type=str, default='./save_cora/GCN',
                        help='Directory with pretrained GCN model_{seed}.pth files')
    parser.add_argument('--save_root', type=str, default=None,
                        help='Directory to save prompt model checkpoints (optional)')
    # Model
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    # Prompt
    parser.add_argument('--prompt_type', type=str, default='dynamic',
                        choices=['dynamic', 'vector', 'CPF', 'CPFplus', 'x'])
    parser.add_argument('--mask_type', type=str, default='adv')
    # Adversarial PGD
    parser.add_argument('--attack_iters', type=int, default=1)
    parser.add_argument('--step_size', type=float, default=0.02)
    # IB (Information Bottleneck) — paper uses IB=True for Cora
    parser.add_argument('--IB', action='store_true', default=False,
                        help='Enable CLUB mutual information regularization')
    parser.add_argument('--w_ib', type=float, default=0.05)
    parser.add_argument('--warm_up', type=int, default=0)
    parser.add_argument('--club_epoch', type=int, default=5)
    parser.add_argument('--club_opt_lr', type=float, default=1e-3)
    parser.add_argument('--hclub', type=int, default=64)
    parser.add_argument('--lclub', type=int, default=1)
    # KNN
    parser.add_argument('--k', type=int, default=10)
    # Evaluation
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--ptb_rates', type=str, default='0.0,0.05,0.1,0.15,0.2,0.25')

    # Cora-specific defaults (matching paper params_M.py)
    parser.add_argument('--ib_norm', type=bool, default=True,
                        help='norm_if in DynamicPrompt: True=no L2 norm (Cora), False=L2 norm (Amazon)')
    parser.add_argument('--lp', type=float, default=0.2)

    args = parser.parse_args()
    ptb_rates = [float(x) for x in args.ptb_rates.split(',')]

    if args.save_root:
        os.makedirs(args.save_root, exist_ok=True)

    # -- Load data --
    adj, adj_f, features, labels, idx_train, idx_val, idx_test, perturbed_adjs = \
        load_cora_metattack(args.data_root, k=args.k, ptb_rates=ptb_rates)

    # -- Verify pretrained models exist --
    for seed in range(args.seeds):
        p = os.path.join(args.pretrain_root, f'model_{seed}.pth')
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing pretrained GCN: {p}. Run train_cora_gcn.py first.")

    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    features = features.to(device)
    adj = adj.to(device)
    adj_f = adj_f.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    perturbed_adjs_dev = {k: v.to(device) for k, v in perturbed_adjs.items()}

    n_class = labels.max().item() + 1
    n_feat = features.shape[1]
    N = features.shape[0]

    all_acc = {ptb: [] for ptb in ptb_rates}
    all_f1  = {ptb: [] for ptb in ptb_rates}

    # -- Per-seed training --
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Build model — norm_if=ib_norm=True for Cora (NO L2 normalization on prompt!)
        model = Node_prompt(
            nfeat=n_feat, nhid=args.hidden, nclass=n_class,
            dropout=args.dropout, p=0.02,
            cuda_id=args.cuda_id,
            IB=args.IB,
            attack_iters=args.attack_iters,
            step_size=args.step_size,
            mask_type=args.mask_type,
            prompt_type=args.prompt_type,
            norm_if=args.ib_norm,   # TRUE for Cora: no L2 normalization
            lp=args.lp,
            num_nodes=N,
        ).to(device)

        # Load pretrained teacher
        state_dict = torch.load(os.path.join(args.pretrain_root, f'model_{seed}.pth'),
                                map_location=device, weights_only=True)
        model.model_teacher.load_state_dict(state_dict)

        # IB: CLUB module + separate optimizer
        club_module, optimizer_club = None, None
        if args.IB:
            from model_prompt1 import CLUB
            club_module = CLUB(n_feat, n_class, args.hclub, args.lclub).to(device)
            optimizer_club = optim.Adam(club_module.parameters(), lr=args.club_opt_lr)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"\n{'='*55}\nSeed {seed}/{args.seeds}\n{'='*55}")

        t0 = time.time()
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            if args.IB:
                output, out_teacher, x_prompt, H_noisy = model(
                    features, adj, adj_f, dataset_str='cora', train=True)
                loss_ib = club_module(x_prompt, H_noisy)
                loss_kl = F.kl_div(output, out_teacher[0], reduction='batchmean', log_target=True)
                loss_train = loss_kl + args.w_ib * loss_ib
                loss_train.backward()
                optimizer.step()
                # CLUB update
                for _ in range(args.club_epoch):
                    club_loss = club_module.learning_loss(x_prompt.detach(), H_noisy.detach())
                    optimizer_club.zero_grad()
                    club_loss.backward()
                    optimizer_club.step()
            else:
                output, out_teacher = model(
                    features, adj, adj_f, dataset_str='cora', train=True)
                loss_train = F.kl_div(output, out_teacher[0], reduction='batchmean', log_target=True)
                loss_train.backward()
                optimizer.step()

            if epoch % 30 == 0 or epoch == args.epochs - 1:
                model.eval()
                with torch.no_grad():
                    out_val = model(features, adj, adj_f, train=False)
                    acc_val = accuracy(out_val[idx_val], labels[idx_val])
                print(f"  ep {epoch:04d}  loss={loss_train.item():.4f}  val_acc={acc_val.item():.4f}")

        train_time = time.time() - t0

        # -- Test on all ptb rates --
        model.eval()
        seed_results = []
        for ptb in ptb_rates:
            adj_test = perturbed_adjs_dev[ptb]
            with torch.no_grad():
                out = model(features, adj_test, adj_f, train=False)
                acc = accuracy(out[idx_test], labels[idx_test]).item()
                f1  = f1_macro(out[idx_test], labels[idx_test]).item()
            all_acc[ptb].append(acc)
            all_f1[ptb].append(f1)
            seed_results.append((ptb, acc, f1))

        print(f"  Train time: {train_time:.0f}s")
        for ptb, acc, f1 in seed_results:
            print(f"    ptb={ptb:.2f}  acc={acc:.4f}  f1={f1:.4f}")

        # Save checkpoint
        if args.save_root:
            ckpt = {
                'seed': seed, 'args': vars(args),
                'generator_state': {k: v.cpu() for k, v in model.generator.state_dict().items()},
            }
            torch.save(ckpt, os.path.join(args.save_root, f'prompt_seed{seed}.pth'))

    # -- Summary --
    print(f"\n{'='*55}")
    print(f"AttrPrompt on Cora 5-shot (Metattack)")
    print(f"Config: prompt={args.prompt_type}, hidden={args.hidden}, epochs={args.epochs}, "
          f"lr={args.lr}, attack_iters={args.attack_iters}, step_size={args.step_size}, "
          f"IB={args.IB}, ib_norm={args.ib_norm}")
    print(f"{'='*55}")
    print(f"{'ptb':>8s}  {'acc_mean':>10s}  {'acc_std':>8s}  {'f1_mean':>10s}  {'f1_std':>8s}")
    print(f"{'-'*48}")

    for ptb in ptb_rates:
        arr = np.array(all_acc[ptb]) * 100
        farr = np.array(all_f1[ptb]) * 100
        print(f"  {ptb:8.2f}  {arr.mean():10.2f}  {arr.std():8.2f}  {farr.mean():10.2f}  {farr.std():8.2f}")
        if ptb == 0.0:
            clean_acc = arr
        if ptb == ptb_rates[-1]:
            robust_acc = arr

    print(f"\n  Clean: {clean_acc.mean():.2f}% ± {clean_acc.std():.2f}%")
    print(f"  M-{ptb_rates[-1]:.2f}: {robust_acc.mean():.2f}% ± {robust_acc.std():.2f}%")


if __name__ == '__main__':
    main()
