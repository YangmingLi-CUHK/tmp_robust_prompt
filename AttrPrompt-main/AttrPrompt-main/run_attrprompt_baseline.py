#!/usr/bin/env python
"""
Batch run AttrPrompt baseline on Cora 5-shot + Metattack.
Runs Phase 1 (GCN pretrain) + Phase 2 (AttrPrompt train) for multiple seeds.

Usage:
    python run_attrprompt_baseline.py --seeds 5 --prompt_type x --epochs_gcn 400 --epochs_prompt 150
"""
import subprocess
import sys
import os
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--prompt_type', type=str, default='x',
                        choices=['dynamic', 'vector', 'CPF', 'CPFplus', 'x'])
    parser.add_argument('--epochs_gcn', type=int, default=400)
    parser.add_argument('--epochs_prompt', type=int, default=150)
    parser.add_argument('--attack_iters', type=int, default=1)
    parser.add_argument('--step_size', type=float, default=0.02)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=16)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Phase 1: Pretrain GCN
    print("="*60)
    print("Phase 1: Pretraining GCN teacher...")
    print("="*60)
    t0 = time.time()
    cmd1 = [
        sys.executable,
        os.path.join(script_dir, 'train_cora_gcn.py'),
        '--seeds', str(args.seeds),
        '--epochs', str(args.epochs_gcn),
        '--lr', str(args.lr),
        '--hidden', str(args.hidden),
    ]
    print(f"Running: {' '.join(cmd1)}")
    subprocess.run(cmd1, check=True)
    print(f"Phase 1 done in {time.time()-t0:.0f}s\n")

    # Phase 2: Train AttrPrompt
    print("="*60)
    print(f"Phase 2: Training AttrPrompt (prompt_type={args.prompt_type})...")
    print("="*60)
    t0 = time.time()
    cmd2 = [
        sys.executable,
        os.path.join(script_dir, 'train_cora_attrprompt.py'),
        '--seeds', str(args.seeds),
        '--epochs', str(args.epochs_prompt),
        '--prompt_type', args.prompt_type,
        '--attack_iters', str(args.attack_iters),
        '--step_size', str(args.step_size),
        '--lr', str(args.lr),
        '--hidden', str(args.hidden),
    ]
    print(f"Running: {' '.join(cmd2)}")
    subprocess.run(cmd2, check=True)
    print(f"Phase 2 done in {time.time()-t0:.0f}s")
    print("\nDone!")

if __name__ == '__main__':
    main()
