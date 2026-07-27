"""
Phase 2: Train AttrPrompt on 5-shot Cora with an explicit base graph.

KL distillation: the prompted/internal-perturbation branch matches the frozen
teacher evaluated on the graph selected by --train_ptb.

With --require_teacher_metadata, the teacher must have been trained on that
exact graph support; rate labels alone are not trusted.

Cora config matches the paper: norm_if=True, IB=True by default.

Usage:
    python train_cora_attrprompt.py --data_root <path> --pretrain_root <path> \
        --train_ptb 0.25 --ptb_rates 0.25 --strict_no_clean_forward
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_prompt1 import Node_prompt
from load_cora_metattack import (
    adjacency_fingerprint,
    load_cora_metattack,
    node_split_fingerprint,
)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    return preds.eq(labels).double().sum() / len(labels)


def f1_macro(output, labels):
    from sklearn.metrics import f1_score
    return f1_score(output.max(1)[1].cpu(), labels.cpu(), average='macro')


def resolve_rate(requested, available):
    matches = [rate for rate in available if np.isclose(rate, requested)]
    if len(matches) != 1:
        raise ValueError(
            f"--train_ptb={requested} must match exactly one rate in "
            f"--ptb_rates={available}")
    return matches[0]


def verify_teacher_checkpoints(pretrain_root, seeds, expected_rate,
                               expected_source, expected_fingerprint,
                               expected_split_source,
                               expected_split_fingerprint,
                               require_metadata, require_no_clean_loaded):
    verified_metadata = {}
    for seed in range(seeds):
        checkpoint_path = os.path.join(
            pretrain_root, f'model_{seed}.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Missing pretrained GCN: {checkpoint_path}. "
                "Run train_cora_gcn.py first.")

        metadata_path = os.path.join(
            pretrain_root, f'model_{seed}.meta.json')
        if not os.path.exists(metadata_path):
            if require_metadata:
                raise FileNotFoundError(
                    f"Missing teacher provenance metadata: {metadata_path}")
            continue

        with open(metadata_path, 'r', encoding='utf-8') as handle:
            metadata = json.load(handle)

        recorded_rate = metadata.get('teacher_train_ptb')
        if recorded_rate is None or not np.isclose(
                float(recorded_rate), expected_rate):
            raise RuntimeError(
                f"Teacher seed {seed} was trained at ptb={recorded_rate}, "
                f"but prompt base is ptb={expected_rate}.")
        if metadata.get('graph_source') != expected_source:
            raise RuntimeError(
                f"Teacher seed {seed} graph source mismatch: "
                f"{metadata.get('graph_source')} != {expected_source}")
        if metadata.get('adjacency_fingerprint') != expected_fingerprint:
            raise RuntimeError(
                f"Teacher seed {seed} adjacency fingerprint does not match "
                "the prompt-training graph.")
        if metadata.get('split_source') != expected_split_source:
            raise RuntimeError(
                f"Teacher seed {seed} node split source mismatch.")
        if metadata.get('split_fingerprint') != expected_split_fingerprint:
            raise RuntimeError(
                f"Teacher seed {seed} node split fingerprint does not match "
                "the prompt-training split.")
        if require_no_clean_loaded and metadata.get(
                'clean_adjacency_loaded') is not False:
            raise RuntimeError(
                f"Teacher seed {seed} metadata does not prove a strict "
                "no-clean-adjacency training process.")
        verified_metadata[seed] = metadata

    if len(verified_metadata) == seeds:
        return verified_metadata, "VERIFIED: teacher trained on this exact base graph"
    return verified_metadata, "UNVERIFIED: legacy checkpoint has no metadata"


def graph_support_stats(adj):
    support = adj > 0
    support = support.clone()
    support.fill_diagonal_(False)
    return int(support.sum().item() // 2)


def structure_sanity(model, features, adj_graph, labels, idx_test):
    """Compare the frozen GCN on the supplied graph versus self-loops only."""
    teacher = model.model_teacher
    teacher.eval()
    identity = torch.eye(adj_graph.shape[0], device=adj_graph.device,
                         dtype=adj_graph.dtype)
    with torch.no_grad():
        hidden_graph = F.relu(teacher.gc1(features, adj_graph))
        hidden_self = F.relu(teacher.gc1(features, identity))
        output_graph = teacher(features, adj_graph)
        output_self = teacher(features, identity)

    result = {
        'graph_acc': accuracy(output_graph[idx_test], labels[idx_test]).item(),
        'self_only_acc': accuracy(output_self[idx_test], labels[idx_test]).item(),
        'hidden_l2_delta': torch.linalg.vector_norm(
            hidden_graph - hidden_self, dim=1).mean().item(),
        'output_abs_delta': (output_graph - output_self).abs().mean().item(),
    }
    del identity, hidden_graph, hidden_self, output_graph, output_self
    return result


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
    parser.add_argument('--ptb_rates', type=str,
                        default='0.0,0.05,0.1,0.15,0.2,0.25',
                        help='Comma-separated graphs to load and evaluate')
    parser.add_argument(
        '--train_ptb', type=float, default=0.0,
        help='Metattack rate used as the prompt-training/reference/validation graph')
    parser.add_argument(
        '--strict_no_clean_forward', action='store_true',
        help=('Require a non-zero train_ptb and non-zero ptb_rates; the clean '
              'adjacency file is not loaded during this Phase-2 process'))
    parser.add_argument(
        '--require_teacher_metadata', action='store_true',
        help=('Require checkpoint sidecars proving that each teacher was '
              'trained on the exact prompt-training adjacency'))

    # Cora-specific defaults (matching paper params_M.py)
    parser.add_argument('--ib_norm', type=bool, default=True,
                        help='norm_if in DynamicPrompt: True=no L2 norm (Cora), False=L2 norm (Amazon)')
    parser.add_argument('--lp', type=float, default=0.2)

    args = parser.parse_args()
    ptb_rates = [float(x) for x in args.ptb_rates.split(',')]
    train_ptb = resolve_rate(args.train_ptb, ptb_rates)

    if args.strict_no_clean_forward:
        if np.isclose(train_ptb, 0.0):
            raise ValueError(
                "--strict_no_clean_forward requires --train_ptb > 0.")
        if any(np.isclose(rate, 0.0) for rate in ptb_rates):
            raise ValueError(
                "--strict_no_clean_forward forbids 0.0 in --ptb_rates.")

    if args.save_root:
        os.makedirs(args.save_root, exist_ok=True)

    # -- Load data --
    adj_clean, adj_f, features, labels, idx_train, idx_val, idx_test, perturbed_adjs = \
        load_cora_metattack(
            args.data_root,
            k=args.k,
            ptb_rates=ptb_rates,
            load_clean_adj=not args.strict_no_clean_forward,
            split_ptb=train_ptb,
        )

    if args.strict_no_clean_forward and adj_clean is not None:
        raise RuntimeError(
            "Protocol violation: clean adjacency tensor was loaded.")

    adj_base_cpu = perturbed_adjs[train_ptb]
    base_source = f"Meta_Self_Cora_{train_ptb}.pt"
    base_fingerprint = adjacency_fingerprint(adj_base_cpu)
    split_source = f"Meta_Self_Cora_{train_ptb}_idx_[train|val|test].npy"
    split_fingerprint = node_split_fingerprint(
        idx_train, idx_val, idx_test)
    teacher_metadata, teacher_provenance = verify_teacher_checkpoints(
        pretrain_root=args.pretrain_root,
        seeds=args.seeds,
        expected_rate=train_ptb,
        expected_source=base_source,
        expected_fingerprint=base_fingerprint,
        expected_split_source=split_source,
        expected_split_fingerprint=split_fingerprint,
        require_metadata=args.require_teacher_metadata,
        require_no_clean_loaded=args.strict_no_clean_forward,
    )
    print("\n" + "=" * 72)
    print("PROMPT-TRAIN GRAPH PROTOCOL")
    print(f"  Base graph source: {base_source}")
    print(f"  Base graph fingerprint: {base_fingerprint}")
    print(f"  Node split source: {split_source}")
    print(f"  Node split fingerprint: {split_fingerprint}")
    print("  Reference branch graph: base graph")
    print("  Prompt/student attack origin: base graph")
    print("  Validation graph: base graph")
    print(f"  Teacher provenance: {teacher_provenance}")
    print("  Teacher weights frozen during Phase 2: YES")
    print("  Node labels in prompt loss: NO")
    print("  Clean adjacency forward during prompt optimization: NO")
    print(f"  Clean adjacency tensor present in process: "
          f"{'YES (evaluation only)' if adj_clean is not None else 'NO'}")
    print(f"  Base graph undirected edge count: "
          f"{graph_support_stats(adj_base_cpu)}")
    if adj_clean is not None and not np.isclose(train_ptb, 0.0):
        clean_support = adj_clean > 0
        base_support = adj_base_cpu > 0
        clean_support.fill_diagonal_(False)
        base_support.fill_diagonal_(False)
        support_difference = int(
            (clean_support != base_support).sum().item() // 2)
        print(f"  Clean/base support symmetric difference: "
              f"{support_difference} undirected edges")
    print("=" * 72 + "\n")

    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    features = features.to(device)
    adj_f = adj_f.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    perturbed_adjs_dev = {k: v.to(device) for k, v in perturbed_adjs.items()}
    adj_base = perturbed_adjs_dev[train_ptb]

    n_class = labels.max().item() + 1
    n_feat = features.shape[1]
    N = features.shape[0]

    all_acc = {ptb: [] for ptb in ptb_rates}
    all_f1  = {ptb: [] for ptb in ptb_rates}
    all_teacher_acc = {ptb: [] for ptb in ptb_rates}
    all_structure = []

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
        if any(parameter.requires_grad
               for parameter in model.model_teacher.parameters()):
            raise RuntimeError(
                "Protocol violation: teacher parameters are trainable.")

        # IB: CLUB module + separate optimizer
        club_module, optimizer_club = None, None
        if args.IB:
            from model_prompt1 import CLUB
            club_module = CLUB(n_feat, n_class, args.hclub, args.lclub).to(device)
            optimizer_club = optim.Adam(club_module.parameters(), lr=args.club_opt_lr)

        optimizer = optim.Adam(
            model.generator.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        print(f"\n{'='*55}\nSeed {seed}/{args.seeds}\n{'='*55}")
        sanity = structure_sanity(
            model, features, adj_base, labels, idx_test)
        all_structure.append(sanity)
        print(
            f"  Frozen teacher @ base: {sanity['graph_acc']:.4f}; "
            f"self-loop-only: {sanity['self_only_acc']:.4f}")
        print(
            f"  Structure check: hidden_L2_delta="
            f"{sanity['hidden_l2_delta']:.6f}, "
            f"output_abs_delta={sanity['output_abs_delta']:.6f}")

        t0 = time.time()
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            if args.IB:
                output, out_teacher, x_prompt, H_noisy = model(
                    features, adj_base, adj_f, dataset_str='cora', train=True)
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
                    features, adj_base, adj_f, dataset_str='cora', train=True)
                loss_train = F.kl_div(output, out_teacher[0], reduction='batchmean', log_target=True)
                loss_train.backward()
                optimizer.step()

            if epoch % 30 == 0 or epoch == args.epochs - 1:
                model.eval()
                with torch.no_grad():
                    out_val = model(features, adj_base, adj_f, train=False)
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
                out_teacher_direct = model.model_teacher(features, adj_test)
                acc = accuracy(out[idx_test], labels[idx_test]).item()
                f1 = float(f1_macro(out[idx_test], labels[idx_test]))
                teacher_acc = accuracy(
                    out_teacher_direct[idx_test], labels[idx_test]).item()
            all_acc[ptb].append(acc)
            all_f1[ptb].append(f1)
            all_teacher_acc[ptb].append(teacher_acc)
            seed_results.append((ptb, teacher_acc, acc, f1))

        print(f"  Train time: {train_time:.0f}s")
        for ptb, teacher_acc, acc, f1 in seed_results:
            print(
                f"    ptb={ptb:.2f}  teacher={teacher_acc:.4f}  "
                f"prompt={acc:.4f}  gain={acc-teacher_acc:+.4f}  "
                f"f1={f1:.4f}")

        # Save checkpoint
        if args.save_root:
            ckpt = {
                'seed': seed, 'args': vars(args),
                'protocol': {
                    'train_ptb': train_ptb,
                    'base_graph_source': base_source,
                    'reference_graph': base_source,
                    'validation_graph': base_source,
                    'adjacency_fingerprint': base_fingerprint,
                    'split_source': split_source,
                    'split_fingerprint': split_fingerprint,
                    'clean_adjacency_loaded': adj_clean is not None,
                    'teacher_train_ptb': (
                        teacher_metadata.get(seed, {})
                        .get('teacher_train_ptb')
                    ),
                    'teacher_provenance': teacher_provenance,
                    'teacher_frozen': True,
                },
                'generator_state': {k: v.cpu() for k, v in model.generator.state_dict().items()},
            }
            torch.save(ckpt, os.path.join(args.save_root, f'prompt_seed{seed}.pth'))

    # -- Summary --
    print(f"\n{'='*82}")
    print("AttrPrompt on Cora 5-shot (Metattack)")
    print(f"Prompt-training base: {base_source}")
    print(f"Strict no-clean-forward: {args.strict_no_clean_forward}")
    print(f"Config: prompt={args.prompt_type}, hidden={args.hidden}, epochs={args.epochs}, "
          f"lr={args.lr}, attack_iters={args.attack_iters}, step_size={args.step_size}, "
          f"IB={args.IB}, ib_norm={args.ib_norm}")
    print(f"{'='*82}")
    print(
        f"{'ptb':>7s}  {'teacher':>10s}  {'prompt':>10s}  "
        f"{'gain':>9s}  {'prompt_std':>10s}  {'f1':>10s}")
    print(f"{'-'*82}")

    rate_summary = {}
    for ptb in ptb_rates:
        arr = np.array(all_acc[ptb]) * 100
        farr = np.array(all_f1[ptb]) * 100
        teacher_arr = np.array(all_teacher_acc[ptb]) * 100
        print(
            f"{ptb:7.2f}  {teacher_arr.mean():10.2f}  {arr.mean():10.2f}  "
            f"{arr.mean()-teacher_arr.mean():+9.2f}  {arr.std():10.2f}  "
            f"{farr.mean():10.2f}")
        rate_summary[str(ptb)] = {
            'teacher_accuracy_mean_pct': float(teacher_arr.mean()),
            'teacher_accuracy_std_pct': float(teacher_arr.std()),
            'prompt_accuracy_mean_pct': float(arr.mean()),
            'prompt_accuracy_std_pct': float(arr.std()),
            'prompt_gain_mean_pct': float(
                arr.mean() - teacher_arr.mean()),
            'prompt_f1_mean_pct': float(farr.mean()),
            'prompt_f1_std_pct': float(farr.std()),
        }

    graph_acc = np.array([item['graph_acc'] for item in all_structure]) * 100
    self_acc = np.array(
        [item['self_only_acc'] for item in all_structure]) * 100
    hidden_delta = np.array(
        [item['hidden_l2_delta'] for item in all_structure])
    output_delta = np.array(
        [item['output_abs_delta'] for item in all_structure])
    print("\nFrozen-teacher structure sanity on the training base:")
    print(f"  graph accuracy: {graph_acc.mean():.2f}% +/- {graph_acc.std():.2f}%")
    print(f"  self-loop-only accuracy: "
          f"{self_acc.mean():.2f}% +/- {self_acc.std():.2f}%")
    print(f"  first-layer embedding L2 delta: {hidden_delta.mean():.6f}")
    print(f"  output mean absolute delta: {output_delta.mean():.6f}")

    if args.save_root:
        summary = {
            'protocol': {
                'prompt_train_ptb': train_ptb,
                'graph_source': base_source,
                'adjacency_fingerprint': base_fingerprint,
                'split_source': split_source,
                'split_fingerprint': split_fingerprint,
                'clean_adjacency_loaded': adj_clean is not None,
                'teacher_provenance': teacher_provenance,
                'teacher_metadata_required': args.require_teacher_metadata,
                'teacher_frozen_during_prompt': True,
            },
            'rates': rate_summary,
            'structure_sanity': {
                'graph_accuracy_mean_pct': float(graph_acc.mean()),
                'graph_accuracy_std_pct': float(graph_acc.std()),
                'self_only_accuracy_mean_pct': float(self_acc.mean()),
                'self_only_accuracy_std_pct': float(self_acc.std()),
                'first_layer_embedding_l2_delta_mean': float(
                    hidden_delta.mean()),
                'output_abs_delta_mean': float(output_delta.mean()),
            },
        }
        with open(os.path.join(args.save_root, 'summary.json'), 'w',
                  encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
