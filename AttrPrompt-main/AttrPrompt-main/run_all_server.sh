#!/bin/bash
# =============================================================================
# AttrPrompt Baseline — Cora 5-shot + Metattack
# Run on remote server: /home/tony/LnL/DFS_HK5
# Environment: conda activate LnL2
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="${PROJECT_ROOT}/data_attack_fewshot/Cora/shot_5/1/Meta_Self/raw"

cd "$SCRIPT_DIR"

echo "============================================"
echo "AttrPrompt Baseline — Cora 5-shot + Metattack"
echo "============================================"
echo "Data: $DATA_ROOT"
echo "Date: $(date)"
echo

# ============================================================================
# Phase 1: Pretrain GCN Teacher (all prompt types share the same teacher)
# ============================================================================
echo ">>> Phase 1: Pretrain GCN Teacher (10 seeds) <<<"
python train_cora_gcn.py \
    --data_root "$DATA_ROOT" \
    --save_root ./save_cora/GCN \
    --epochs 400 \
    --hidden 16 \
    --lr 0.01 \
    --dropout 0.5 \
    --weight_decay 5e-4 \
    --patience 100 \
    --seeds 10 \
    --cuda_id 0

echo
echo ">>> Phase 1 Complete <<<"
echo

# ============================================================================
# Phase 2a: AttrPrompt dynamic (paper default)
# ============================================================================
echo ">>> Phase 2a: AttrPrompt dynamic (paper default config) <<<"
python train_cora_attrprompt.py \
    --data_root "$DATA_ROOT" \
    --pretrain_root ./save_cora/GCN \
    --save_root ./save_cora/AttrPrompt_dynamic \
    --prompt_type dynamic \
    --ib_norm True \
    --IB \
    --epochs 200 \
    --hidden 16 \
    --lr 0.01 \
    --attack_iters 1 \
    --step_size 0.02 \
    --seeds 10 \
    --cuda_id 0

echo
echo ">>> Phase 2a Complete <<<"
echo

# ============================================================================
# Phase 2b: AttrPrompt x (simple linear, best in our local test)
# ============================================================================
echo ">>> Phase 2b: AttrPrompt x (linear feature transform) <<<"
python train_cora_attrprompt.py \
    --data_root "$DATA_ROOT" \
    --pretrain_root ./save_cora/GCN \
    --save_root ./save_cora/AttrPrompt_x \
    --prompt_type x \
    --ib_norm True \
    --epochs 150 \
    --hidden 16 \
    --lr 0.01 \
    --attack_iters 1 \
    --step_size 0.02 \
    --seeds 10 \
    --cuda_id 0

echo
echo ">>> Phase 2b Complete <<<"
echo

# ============================================================================
# Phase 2c (optional): GCN teacher direct test on all ptb rates (no prompt)
# ============================================================================
echo ">>> Phase 2c: GCN teacher direct test (no prompt baseline) <<<"
python -c "
import sys, os, torch, numpy as np
sys.path.insert(0, '$SCRIPT_DIR')
from model_prompt1 import GCN
from load_cora_metattack import load_cora_metattack

adj, adj_f, features, labels, idx_train, idx_val, idx_test, perturbed = \
    load_cora_metattack('$DATA_ROOT', k=10)

device = torch.device('cuda:0')
features, labels = features.to(device), labels.to(device)
idx_test = idx_test.to(device)
perturbed = {k: v.to(device) for k, v in perturbed.items()}
n_class = labels.max().item() + 1

ptb_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
all_acc = {p: [] for p in ptb_rates}

for seed in range(10):
    model = GCN(features.shape[1], 16, n_class, 0.5, 0.02, 0).to(device)
    sd = torch.load(f'./save_cora/GCN/model_{seed}.pth', map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        for ptb in ptb_rates:
            out = model(features, perturbed[ptb], 0, test=1)
            preds = out.max(1)[1]
            acc = (preds[idx_test] == labels[idx_test]).float().mean().item()
            all_acc[ptb].append(acc)

print()
print('GCN Teacher direct test (no prompt):')
for ptb in ptb_rates:
    arr = np.array(all_acc[ptb]) * 100
    print(f'  ptb={ptb:.2f}: {arr.mean():.2f}% ± {arr.std():.2f}%')
print()
"

echo
echo "============================================"
echo "All experiments complete."
echo "Date: $(date)"
echo "============================================"
