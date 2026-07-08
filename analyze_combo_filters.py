#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, pstdev


NAME_RE = re.compile(
    r"^(?P<bb>stable|peak)_(?P<tag>.+?)_sim(?P<sim>-?\d+(?:\.\d+)?)_"
    r"deg(?P<deg>-?\d+(?:\.\d+)?)_ood(?P<ood>-?\d+(?:\.\d+)?)_"
    r"ptb(?P<ptb>\d+(?:\.\d+)?)\.log$"
)
SEED_RE = re.compile(r"(?:seed:\s*(\d+)\s*\|\s*split\s*\d+\s*:|split:\s*\d+\s*\|\s*seed\s*(\d+)\s*:)\s*([-+]?nan|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.I)
FINAL_RE = re.compile(r"Final True Accuracy:\s*([-+]?nan|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.I)
AGG_RE = re.compile(r"# Split\s+\d+\s+Muti Seed Acc without min value:\s*([-+]?nan|[-+]?\d*\.?\d+)(?:\s*±\s*([-+]?nan|[-+]?\d*\.?\d+))?", re.I)

DETECTION_FIELDS = [
    "pollution_added_edges",
    "pollution_deleted_edges",
    "pollution_clean_edges",
    "pollution_attack_edges",
    "filter_module_f1",
    "filter_module_tpr",
    "filter_module_tnr",
    "filter_module_balanced_accuracy",
    "gppt_filter_f1",
    "gppt_filter_tpr",
    "gppt_filter_tnr",
    "gppt_filter_balanced_accuracy",
    "tau_tune_f1",
    "tau_tune_tpr",
    "tau_tune_tnr",
    "tau_tune_balanced_accuracy",
    "out_detect_pt_edges_f1",
    "out_detect_pt_edges_tpr",
    "out_detect_pt_edges_tnr",
    "out_detect_pt_edges_balanced_accuracy",
    "sim_pt_node_f1",
    "sim_pt_node_tpr",
    "sim_pt_node_tnr",
    "sim_pt_node_balanced_accuracy",
    "sim_pt_edge_f1",
    "sim_pt_edge_tpr",
    "sim_pt_edge_tnr",
    "sim_pt_edge_balanced_accuracy",
    "degree_pt_node_f1",
    "degree_pt_node_tpr",
    "degree_pt_node_tnr",
    "degree_pt_node_balanced_accuracy",
    "degree_pt_edge_f1",
    "degree_pt_edge_tpr",
    "degree_pt_edge_tnr",
    "degree_pt_edge_balanced_accuracy",
    "out_detect_pt_node_f1",
    "out_detect_pt_node_tpr",
    "out_detect_pt_node_tnr",
    "out_detect_pt_node_balanced_accuracy",
    "out_detect_pt_edge_f1",
    "out_detect_pt_edge_tpr",
    "out_detect_pt_edge_tnr",
    "out_detect_pt_edge_balanced_accuracy",
]


def parse_float(value):
    if value is None:
        return None
    value = value.strip()
    if value.lower() == "nan":
        return math.nan
    return float(value)


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{value:.4f}"


def finite(values):
    return [v for v in values if v is not None and not math.isnan(v)]


def parse_key_value_line(line):
    values = {}
    for part in line.split("|")[1:]:
        part = part.strip()
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def parse_numeric(value):
    if value is None:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return parse_float(value)
    except ValueError:
        return value


def mean_field(rows, source, field):
    values = finite([
        parse_numeric(row.get(field))
        for row in rows
        if row.get("source") == source and isinstance(parse_numeric(row.get(field)), float)
    ])
    return mean(values) if values else None


def mean_tip_field(rows, tip, field):
    values = finite([
        parse_numeric(row.get(field))
        for row in rows
        if row.get("tip") == tip and isinstance(parse_numeric(row.get(field)), float)
    ])
    return mean(values) if values else None


def parse_detection_metrics(text):
    edge_rows = []
    tip_rows = []
    pollution = {}
    for line in text.splitlines():
        if line.startswith("Pollution Diff |"):
            values = parse_key_value_line(line)
            for key in ["added_edges", "deleted_edges", "clean_edges", "attack_edges"]:
                parsed = parse_numeric(values.get(key))
                if isinstance(parsed, float):
                    pollution[key] = parsed
        elif line.startswith("Edge Detection |") and "skipped=true" not in line:
            edge_rows.append(parse_key_value_line(line))
        elif line.startswith("Tip Detection |"):
            tip_rows.append(parse_key_value_line(line))

    out = {
        "pollution_added_edges": pollution.get("added_edges"),
        "pollution_deleted_edges": pollution.get("deleted_edges"),
        "pollution_clean_edges": pollution.get("clean_edges"),
        "pollution_attack_edges": pollution.get("attack_edges"),
    }
    for source in ["filter_module", "gppt_filter", "tau_tune", "out_detect_pt_edges"]:
        out[f"{source}_f1"] = mean_field(edge_rows, source, "f1")
        out[f"{source}_tpr"] = mean_field(edge_rows, source, "tpr")
        out[f"{source}_tnr"] = mean_field(edge_rows, source, "tnr")
        out[f"{source}_balanced_accuracy"] = mean_field(edge_rows, source, "balanced_accuracy")
    for tip in ["sim_pt", "degree_pt", "out_detect_pt"]:
        out[f"{tip}_node_f1"] = mean_tip_field(tip_rows, tip, "node_f1")
        out[f"{tip}_node_tpr"] = mean_tip_field(tip_rows, tip, "node_tpr")
        out[f"{tip}_node_tnr"] = mean_tip_field(tip_rows, tip, "node_tnr")
        out[f"{tip}_node_balanced_accuracy"] = mean_tip_field(tip_rows, tip, "node_balanced_accuracy")
        out[f"{tip}_edge_f1"] = mean_tip_field(tip_rows, tip, "edge_f1")
        out[f"{tip}_edge_tpr"] = mean_tip_field(tip_rows, tip, "edge_tpr")
        out[f"{tip}_edge_tnr"] = mean_tip_field(tip_rows, tip, "edge_tnr")
        out[f"{tip}_edge_balanced_accuracy"] = mean_tip_field(tip_rows, tip, "edge_balanced_accuracy")
    return out


def parse_log(path):
    meta = NAME_RE.match(path.name)
    if not meta:
        return None

    text = path.read_text(errors="replace")
    seed_values = {}
    for match in SEED_RE.finditer(text):
        seed = int(match.group(1) or match.group(2))
        seed_values[seed] = parse_float(match.group(3))

    finals = [parse_float(m.group(1)) for m in FINAL_RE.finditer(text)]
    if not seed_values and finals:
        seed_values = {idx + 1: value for idx, value in enumerate(finals)}

    values = [seed_values[s] for s in sorted(seed_values)]
    valid = finite(values)
    trimmed = valid[:]
    if len(trimmed) > 1:
        trimmed.remove(min(trimmed))

    agg_match = AGG_RE.search(text)
    reported_mean = parse_float(agg_match.group(1)) if agg_match else None
    reported_std = parse_float(agg_match.group(2)) if agg_match and agg_match.group(2) else None

    row = meta.groupdict()
    detection_metrics = parse_detection_metrics(text)
    row.update(
        {
            "file": str(path),
            "num_seeds": len(values),
            "valid_seeds": len(valid),
            "has_nan": "nan" in text.lower(),
            "has_traceback": "Traceback" in text,
            "complete": len(values) >= 5 and not ("Traceback" in text),
            "mean": mean(valid) if valid else math.nan,
            "std": pstdev(valid) if len(valid) > 1 else 0.0 if valid else math.nan,
            "trim_mean_without_min": mean(trimmed) if trimmed else math.nan,
            "trim_std_without_min": pstdev(trimmed) if len(trimmed) > 1 else 0.0 if trimmed else math.nan,
            "reported_mean_without_min": reported_mean,
            "reported_std_without_min": reported_std,
        }
    )
    row.update(detection_metrics)
    for seed in range(1, 6):
        row[f"seed{seed}"] = seed_values.get(seed)
    return row


def sort_key(row):
    return (
        row["bb"],
        float(row["ptb"]),
        row["tag"],
        float(row["sim"]),
        float(row["deg"]),
        float(row["ood"]),
    )


def write_csv(rows, path):
    fields = [
        "bb",
        "tag",
        "ptb",
        "sim",
        "deg",
        "ood",
        "valid_seeds",
        "num_seeds",
        "complete",
        "has_nan",
        "has_traceback",
        "mean",
        "std",
        "trim_mean_without_min",
        "trim_std_without_min",
        "reported_mean_without_min",
        "reported_std_without_min",
        *DETECTION_FIELDS,
        "seed1",
        "seed2",
        "seed3",
        "seed4",
        "seed5",
        "file",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            for key, value in out.items():
                if isinstance(value, float):
                    out[key] = fmt(value)
            writer.writerow(out)


def markdown_table(rows):
    header = "| BB | ptb | combo | sim | deg | ood | valid | mean | mean w/o min | seeds | flags |"
    sep = "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|"
    lines = [header, sep]
    for row in rows:
        seeds = ", ".join(fmt(row.get(f"seed{i}")) or "-" for i in range(1, 6))
        flags = []
        if row["has_nan"]:
            flags.append("NaN")
        if row["has_traceback"]:
            flags.append("Traceback")
        if not row["complete"]:
            flags.append("incomplete")
        lines.append(
            f"| {row['bb']} | {row['ptb']} | {row['tag']} | {row['sim']} | {row['deg']} | {row['ood']} | "
            f"{row['valid_seeds']}/5 | {fmt(row['mean'])} | {fmt(row['trim_mean_without_min'])} | {seeds} | {', '.join(flags) or '-'} |"
        )
    return "\n".join(lines)


def write_markdown(rows, path):
    complete_rows = [r for r in rows if r["complete"] and r["valid_seeds"] > 0]
    best_by_ptb = {}
    for row in complete_rows:
        key = (row["bb"], row["ptb"])
        if key not in best_by_ptb or row["trim_mean_without_min"] > best_by_ptb[key]["trim_mean_without_min"]:
            best_by_ptb[key] = row

    best_overall = sorted(
        complete_rows,
        key=lambda r: (r["trim_mean_without_min"], r["mean"]),
        reverse=True,
    )[:10]

    lines = [
        "# combo_filters Summary",
        "",
        f"- Parsed logs: {len(rows)}",
        f"- Complete logs: {sum(1 for r in rows if r['complete'])}",
        f"- Logs with NaN text: {sum(1 for r in rows if r['has_nan'])}",
        f"- Logs with Traceback: {sum(1 for r in rows if r['has_traceback'])}",
        "",
        "## Best Per Backbone And Ptb",
        "",
        markdown_table([best_by_ptb[k] for k in sorted(best_by_ptb, key=lambda x: (x[0], float(x[1])))]),
        "",
        "## Top 10 Overall",
        "",
        markdown_table(best_overall),
        "",
        "## Full Table",
        "",
        markdown_table(rows),
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs/combo_filters")
    parser.add_argument("--out-prefix", default="logs/combo_filters/combo_filters_summary")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"Log directory not found: {log_dir}")

    rows = []
    skipped = []
    for path in sorted(log_dir.glob("*.log")):
        row = parse_log(path)
        if row is None:
            skipped.append(path.name)
        else:
            rows.append(row)

    rows.sort(key=sort_key)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_prefix.with_suffix(".csv"))
    write_markdown(rows, out_prefix.with_suffix(".md"))

    print(f"Parsed logs: {len(rows)}")
    if skipped:
        print("Skipped non-matching logs:", ", ".join(skipped))
    print(f"Wrote: {out_prefix.with_suffix('.csv')}")
    print(f"Wrote: {out_prefix.with_suffix('.md')}")

    complete = [r for r in rows if r["complete"] and r["valid_seeds"] > 0]
    if complete:
        best = max(complete, key=lambda r: (r["trim_mean_without_min"], r["mean"]))
        print(
            "Best overall:",
            best["bb"],
            best["tag"],
            f"ptb={best['ptb']}",
            f"sim={best['sim']}",
            f"deg={best['deg']}",
            f"ood={best['ood']}",
            f"mean_wo_min={fmt(best['trim_mean_without_min'])}",
            f"mean={fmt(best['mean'])}",
        )


if __name__ == "__main__":
    main()
