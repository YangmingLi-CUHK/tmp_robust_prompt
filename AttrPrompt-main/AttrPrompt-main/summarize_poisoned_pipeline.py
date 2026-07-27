"""Aggregate the diagonal poisoned-teacher AttrPrompt sweep into one CSV."""
import argparse
import csv
import json
import os
from rate_utils import canonical_rate, canonical_rate_tokens, rate_tag


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing experiment summary: {path}")
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--ptb_rates', nargs='+', required=True)
    parser.add_argument('--prompt_type', default='dynamic')
    parser.add_argument('--allow_overwrite', action='store_true')
    parser.add_argument(
        '--mode', choices=['all', 'per-rate', 'combined'], default='all')
    args = parser.parse_args()

    rate_tokens = canonical_rate_tokens(args.ptb_rates)
    rows = []
    rate_roots = []
    for rate_text in rate_tokens:
        rate = canonical_rate(rate_text)
        tag = rate_tag(rate_text)
        rate_root = os.path.join(args.output_root, tag)
        rate_roots.append(rate_root)
        teacher = load_json(os.path.join(
            rate_root, 'GCN', 'summary.json'))
        prompt = load_json(os.path.join(
            rate_root, f'AttrPrompt_{args.prompt_type}', 'summary.json'))

        teacher_protocol = teacher['protocol']
        prompt_protocol = prompt['protocol']
        if (teacher_protocol['adjacency_fingerprint'] !=
                prompt_protocol['adjacency_fingerprint']):
            raise RuntimeError(
                f"M-{rate_text}: teacher/prompt graph fingerprints differ.")
        if (teacher_protocol['split_fingerprint'] !=
                prompt_protocol['split_fingerprint']):
            raise RuntimeError(
                f"M-{rate_text}: teacher/prompt node split fingerprints differ.")

        rate_metrics = prompt['rates'][rate_text]
        sanity = prompt['structure_sanity']
        teacher_train_acc = teacher['metrics']['test_accuracy_mean_pct']
        teacher_direct_acc = rate_metrics['teacher_accuracy_mean_pct']
        if abs(teacher_train_acc - teacher_direct_acc) > 1e-5:
            raise RuntimeError(
                f"M-{rate_text}: saved teacher accuracy changed between "
                f"Phase 1 ({teacher_train_acc}) and frozen Phase 2 "
                f"({teacher_direct_acc}).")
        rows.append({
            'ptb_rate': rate_text,
            'teacher_train_test_acc_pct': teacher_train_acc,
            'teacher_phase2_direct_acc_pct': teacher_direct_acc,
            'attrprompt_acc_pct': (
                rate_metrics['prompt_accuracy_mean_pct']),
            'attrprompt_acc_std_pct': (
                rate_metrics['prompt_accuracy_std_pct']),
            'prompt_gain_pct': rate_metrics['prompt_gain_mean_pct'],
            'attrprompt_f1_pct': rate_metrics['prompt_f1_mean_pct'],
            'self_only_teacher_acc_pct': (
                sanity['self_only_accuracy_mean_pct']),
            'embedding_l2_graph_vs_self': (
                sanity['first_layer_embedding_l2_delta_mean']),
            'output_abs_graph_vs_self': sanity['output_abs_delta_mean'],
            'adjacency_fingerprint': (
                teacher_protocol['adjacency_fingerprint']),
        })

    fieldnames = list(rows[0].keys())
    rate_output_paths = []
    if args.mode in ('all', 'per-rate'):
        for row, rate_root in zip(rows, rate_roots):
            rate_output_path = os.path.join(
                rate_root, f"result_{rate_tag(row['ptb_rate'])}.csv")
            if os.path.exists(rate_output_path) and not args.allow_overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite per-rate CSV: {rate_output_path}")
            with open(rate_output_path, 'w', newline='',
                      encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            rate_output_paths.append(rate_output_path)

    output_path = None
    if args.mode in ('all', 'combined'):
        output_path = os.path.join(
            args.output_root, 'poisoned_pipeline_summary.csv')
        if os.path.exists(output_path) and not args.allow_overwrite:
            raise FileExistsError(
                f"Refusing to overwrite combined CSV: {output_path}")
        with open(output_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("\nCombined poisoned-teacher AttrPrompt results")
    print(
        f"{'ptb':>6s}  {'teacher':>9s}  {'prompt':>9s}  "
        f"{'gain':>8s}  {'f1':>9s}  {'self-only':>10s}")
    print("-" * 62)
    for row in rows:
        print(
            f"{float(row['ptb_rate']):6.2f}  "
            f"{row['teacher_phase2_direct_acc_pct']:9.2f}  "
            f"{row['attrprompt_acc_pct']:9.2f}  "
            f"{row['prompt_gain_pct']:+8.2f}  "
            f"{row['attrprompt_f1_pct']:9.2f}  "
            f"{row['self_only_teacher_acc_pct']:10.2f}")
    for rate_output_path in rate_output_paths:
        print(f"Saved per-rate: {rate_output_path}")
    if output_path:
        print(f"Saved combined: {output_path}")


if __name__ == '__main__':
    main()
