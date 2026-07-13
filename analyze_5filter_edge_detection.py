#!/usr/bin/env python3
"""Parse per-seed combo edge-detection metrics and write tidy CSV summaries."""

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, pstdev


LOG_NAME_RE = re.compile(r"^(?P<method>.+)_ptb(?P<ptb>\d+(?:\.\d+)?)\.log$")
METRIC_PREFIX = "Combo Edge Detection |"
METRICS = [
    "f1", "auc", "ap", "best_f1", "best_threshold", "precision", "recall",
    "tpr", "tnr", "tp", "fp", "fn", "tn",
]


def parse_value(value):
    value = value.strip()
    if value.lower() == "nan":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return value


def parse_key_values(line):
    values = {}
    for part in line.split("|")[1:]:
        if "=" not in part:
            continue
        key, value = part.strip().split("=", 1)
        values[key.strip()] = parse_value(value)
    return values


def parse_log(path):
    match = LOG_NAME_RE.match(path.name)
    if match is None:
        return []

    file_method = match.group("method")
    ptb = float(match.group("ptb"))
    rows = []
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        if not line.startswith(METRIC_PREFIX):
            continue
        values = parse_key_values(line)
        row = {
            "method": str(values.get("method", file_method)),
            "num_filters": int(values.get("num_filters", file_method.count("+") + 1)),
            "ptb": ptb,
            "seed": len(rows) + 1,
            "fusion": str(values.get("fusion", "unknown")),
            "file": str(path),
        }
        for metric in METRICS:
            row[metric] = values.get(metric, math.nan)
        rows.append(row)
    return rows


def finite(values):
    return [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]


def aggregate(rows):
    grouped = {}
    for row in rows:
        key = (row["method"], row["num_filters"], row["ptb"], row["fusion"])
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (method, num_filters, ptb, fusion), group in grouped.items():
        summary = {
            "method": method,
            "num_filters": num_filters,
            "ptb": ptb,
            "fusion": fusion,
            "num_seeds": len(group),
            "complete": len(group) >= 5,
        }
        for metric in METRICS:
            values = finite(row[metric] for row in group)
            summary[f"{metric}_mean"] = mean(values) if values else math.nan
            summary[f"{metric}_std"] = pstdev(values) if len(values) > 1 else (0.0 if values else math.nan)
        summaries.append(summary)
    return sorted(summaries, key=lambda row: (row["num_filters"], row["method"], row["ptb"]))


def format_cell(value):
    if isinstance(value, float):
        return "NaN" if math.isnan(value) else f"{value:.6f}"
    return value


def write_csv(rows, path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_cell(row.get(field, "")) for field in fields})


def write_markdown(rows, path):
    lines = [
        "# Five-filter edge anomaly detection summary",
        "",
        "Fusion: binary union for fixed-threshold F1; rank-normalized maximum score for AUC/AP.",
        "",
        "| filters | method | ptb | seeds | AUC | F1 | AP |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['num_filters']} | {row['method']} | {row['ptb']:.2f} | {row['num_seeds']} | "
            f"{format_cell(row['auc_mean'])} ± {format_cell(row['auc_std'])} | "
            f"{format_cell(row['f1_mean'])} ± {format_cell(row['f1_std'])} | "
            f"{format_cell(row['ap_mean'])} ± {format_cell(row['ap_std'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs/5filter_edge_detection")
    parser.add_argument("--out-dir", default="results/5filter_edge_detection")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"Log directory not found: {log_dir}")

    rows = []
    for path in sorted(log_dir.glob("*.log")):
        rows.extend(parse_log(path))
    rows.sort(key=lambda row: (row["num_filters"], row["method"], row["ptb"], row["seed"]))
    if not rows:
        raise SystemExit(f"No '{METRIC_PREFIX}' records found in {log_dir}")

    summaries = aggregate(rows)
    out_dir = Path(args.out_dir)
    per_seed_fields = ["method", "num_filters", "ptb", "seed", "fusion", *METRICS, "file"]
    summary_fields = [
        "method", "num_filters", "ptb", "fusion", "num_seeds", "complete",
        *[field for metric in METRICS for field in (f"{metric}_mean", f"{metric}_std")],
    ]
    write_csv(rows, out_dir / "per_seed_metrics.csv", per_seed_fields)
    write_csv(summaries, out_dir / "summary.csv", summary_fields)
    write_markdown(summaries, out_dir / "summary.md")
    print(f"Parsed {len(rows)} seed records from {log_dir}")
    print(f"Wrote {out_dir / 'per_seed_metrics.csv'}")
    print(f"Wrote {out_dir / 'summary.csv'}")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
