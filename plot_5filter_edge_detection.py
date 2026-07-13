#!/usr/bin/env python3
"""Plot AUC/F1 against attack concentration for 1/2/3-filter methods."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def number(value):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else math.nan
    except (TypeError, ValueError):
        return math.nan


def read_summary(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["num_filters"] = int(row["num_filters"])
        row["ptb"] = number(row["ptb"])
    return rows


def plot_metric(rows, metric, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharex=True, sharey=True)
    labels = {1: "Single filter", 2: "Two-filter combinations", 3: "Three-filter combinations"}

    for num_filters, ax in zip((1, 2, 3), axes):
        panel = [row for row in rows if row["num_filters"] == num_filters]
        methods = sorted({row["method"] for row in panel})
        for method in methods:
            method_rows = sorted(
                (row for row in panel if row["method"] == method),
                key=lambda row: row["ptb"],
            )
            x = [row["ptb"] for row in method_rows]
            y = [number(row[f"{metric}_mean"]) for row in method_rows]
            std = [number(row[f"{metric}_std"]) for row in method_rows]
            ax.plot(x, y, marker="o", linewidth=1.7, markersize=4, label=method)
            lower = [value - error for value, error in zip(y, std)]
            upper = [value + error for value, error in zip(y, std)]
            ax.fill_between(x, lower, upper, alpha=0.10)

        ax.set_title(labels[num_filters])
        ax.set_xlabel("Attack concentration")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, frameon=False, ncol=1 if num_filters == 1 else 2)

    axes[0].set_ylabel("ROC-AUC" if metric == "auc" else "F1 score")
    fig.suptitle(f"Edge anomaly detection: {'ROC-AUC' if metric == 'auc' else 'F1 score'}", fontsize=15)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"edge_detection_{metric}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/5filter_edge_detection/summary.csv")
    parser.add_argument("--out-dir", default="results/5filter_edge_detection/figures")
    args = parser.parse_args()

    summary = Path(args.summary)
    if not summary.exists():
        raise SystemExit(f"Summary CSV not found: {summary}")
    rows = read_summary(summary)
    rows = [row for row in rows if row.get("complete", "").lower() == "true"]
    if not rows:
        raise SystemExit("No complete five-seed rows found in summary CSV")
    out_dir = Path(args.out_dir)
    plot_metric(rows, "auc", out_dir)
    plot_metric(rows, "f1", out_dir)
    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
