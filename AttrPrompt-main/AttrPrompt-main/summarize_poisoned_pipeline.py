"""Aggregate the diagonal poisoned-teacher AttrPrompt sweep into one CSV."""
import argparse
import csv
import json
import os


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
    args = parser.parse_args()

    rows = []
    for rate_text in args.ptb_rates:
        rate = float(rate_text)
        tag = rate_text.replace('.', 'p')
        rate_root = os.path.join(args.output_root, f'M{tag}')
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

        rate_metrics = prompt['rates'][str(rate)]
        sanity = prompt['structure_sanity']
        teacher_train_acc = teacher['metrics']['test_accuracy_mean_pct']
        teacher_direct_acc = rate_metrics['teacher_accuracy_mean_pct']
        if abs(teacher_train_acc - teacher_direct_acc) > 1e-5:
            raise RuntimeError(
                f"M-{rate_text}: saved teacher accuracy changed between "
                f"Phase 1 ({teacher_train_acc}) and frozen Phase 2 "
                f"({teacher_direct_acc}).")
        rows.append({
            'ptb_rate': rate,
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

    output_path = os.path.join(
        args.output_root, 'poisoned_pipeline_summary.csv')
    fieldnames = list(rows[0].keys())
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
            f"{row['ptb_rate']:6.2f}  "
            f"{row['teacher_phase2_direct_acc_pct']:9.2f}  "
            f"{row['attrprompt_acc_pct']:9.2f}  "
            f"{row['prompt_gain_pct']:+8.2f}  "
            f"{row['attrprompt_f1_pct']:9.2f}  "
            f"{row['self_only_teacher_acc_pct']:10.2f}")
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
