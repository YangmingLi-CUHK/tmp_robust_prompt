#!/usr/bin/env python3
"""
Meeting 15 报告生成器 v3 — 极简版，全量数据，直接呈现

修正：
- GCL clean graph (ptb=0.0) 从 eval_pretrain 日志提取
- 所有 accuracy 以 mean±std 呈现，5 seed 全部计入（不除最小值）
- 中间结果只看边异常检测：TPR, TNR, AUC, F1
- peak backbone only
"""
import os
import re
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

BASE_DIR = Path(r"c:\Users\80406\Desktop\DFS_HK5")
LOGS_DIR = BASE_DIR / "logs"
OUTPUT_HTML = BASE_DIR / "reports" / "meeting15_report.html"

PTB_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
BB = "peak"

# ============================================================
# UTILS
# ============================================================

def parse_seeds_from_log(filepath):
    """
    Read a training log file. Return:
      seeds: dict {seed_num: acc}  (one entry per seed found)
    Uses per-seed 'Final True Accuracy' lines (one per seed within the file).
    """
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    # Pattern:  "# Seed N Muti Split Final Acc: X.XXXX"
    seeds = {}
    for m in re.finditer(r'# Seed (\d+) Muti Split Final Acc: ([\d.]+)', content):
        seeds[int(m.group(1))] = float(m.group(2))
    return seeds


def seeds_to_stat(seeds):
    """Return (mean, std, n_valid, all_values_str). Mean includes ALL seeds."""
    vals = [v for v in seeds.values() if not (isinstance(v, float) and v != v)]
    if not vals:
        return None, None, 0, ""
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
    vals_str = ", ".join(f"{v:.4f}" for v in sorted(vals))
    return mean, std, len(vals), vals_str


# ============================================================
# SINGLE FILTER DATA
# ============================================================

FILTER_DIRS = {
    "sim": LOGS_DIR / "single_filter_sim",
    "degree": LOGS_DIR / "single_filter_degree",
    "ood": LOGS_DIR / "single_filter_ood",
}

def collect_single_filter():
    """Return data[filter_type][threshold][ptb] = {seeds, mean, std, ...}"""
    data = defaultdict(lambda: defaultdict(dict))
    for ftype, dirpath in FILTER_DIRS.items():
        if not dirpath.exists():
            continue
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith('.log'):
                continue
            parts = fname.split('_')
            # filename: sim0.3_peak_0.0_... or deg1_peak_0.0_... or ood0.5_peak_0.0_...
            if ftype == "sim":
                thr = float(parts[0].replace('sim', ''))
            elif ftype == "degree":
                thr = int(parts[0].replace('deg', ''))
            else:
                thr = float(parts[0].replace('ood', ''))
            file_bb = parts[1]
            if file_bb != BB:
                continue
            ptb = float(parts[2])
            seeds = parse_seeds_from_log(dirpath / fname)
            if seeds:
                mean, std, n, vals_str = seeds_to_stat(seeds)
                data[ftype][thr][ptb] = {"seeds": seeds, "mean": mean, "std": std, "n": n, "seeds_str": vals_str}
    return data


# ============================================================
# BASELINE DATA (GPPT + GCL)
# ============================================================

def collect_gppt():
    """Return data[ptb] = {seeds, mean, std, ...}"""
    data = {}
    d = LOGS_DIR / "baselines"
    if not d.exists():
        return data
    for fname in sorted(os.listdir(d)):
        if not fname.endswith('.log') or not fname.startswith(f'gppt_{BB}'):
            continue
        # gppt_peak_0.0_...
        parts = fname.split('_')
        ptb = float(parts[2])
        seeds = parse_seeds_from_log(d / fname)
        if seeds:
            mean, std, n, vals_str = seeds_to_stat(seeds)
            data[ptb] = {"seeds": seeds, "mean": mean, "std": std, "n": n, "seeds_str": vals_str}
    return data


def collect_gcl():
    """
    GCL Linear Probe.
    - Clean (ptb=0.0): from eval_pretrain log (single checkpoint, no seed variation)
    - Attacked (ptb>=0.05): from baselines/gcl_lp_peak_*.log (single evaluation per file)
    """
    data = {}

    # --- Clean graph ---
    eval_log = LOGS_DIR / "eval_pretrain_20260619_215608.log"
    if eval_log.exists():
        with open(eval_log, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        # Peak checkpoint: permE/maskN, lr=0.001, ratio=0.3, seed=1
        m = re.search(
            r'aug1_permE\.aug2_maskN\.lr_0\.001\.ratio_0\.3\.seed_1\.pth.*?Test:\s*([\d.]+)',
            content
        )
        if m:
            acc = float(m.group(1))
            data[0.0] = {"seeds": {1: acc}, "mean": acc, "std": 0.0, "n": 1, "seeds_str": f"{acc:.4f}"}

    # --- Attacked graphs ---
    d = LOGS_DIR / "baselines"
    if d.exists():
        for fname in sorted(os.listdir(d)):
            if not fname.endswith('.log') or not fname.startswith(f'gcl_lp_{BB}'):
                continue
            # gcl_lp_peak_0.05_...
            parts = fname.split('_')
            ptb = float(parts[3])
            with open(d / fname, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            m = re.search(r'BEST CHECKPOINT:.*?\n\s+Test Acc:\s*([\d.]+)', content)
            if m:
                acc = float(m.group(1))
                data[ptb] = {"seeds": {1: acc}, "mean": acc, "std": 0.0, "n": 1, "seeds_str": f"{acc:.4f}"}

    return data


# ============================================================
# COMBO FILTER DATA  (both directories)
# ============================================================

def collect_combos():
    """
    Parse both combo_filters/ and combo_filters_midmetrics/.
    Return data[label][ptb] = {seeds, mean, std, ...}
    """
    data = {}

    for subdir in ["combo_filters", "combo_filters_midmetrics"]:
        d = LOGS_DIR / subdir
        if not d.exists():
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith('.log') or not fname.startswith(BB):
                continue
            fpath = d / fname
            stem = fname.replace('.log', '')

            # Parse combo type and thresholds
            # combo_filters:       peak_deg+ood_sim-1.0_deg1_ood0.5_ptb0.0.log
            # combo_filters_mid:   peak_deg_ood_sim-1.0_deg1_ood0.5_ptb0.0.log
            #                       peak_all3_sim0.3_deg1_ood0.5_ptb0.0.log

            if 'all3' in stem:
                combo = 'all3'
            elif 'deg+ood' in stem or 'deg_ood' in stem:
                combo = 'deg+ood'
            elif 'sim+deg' in stem or 'sim_deg' in stem:
                combo = 'sim+deg'
            elif 'sim+ood' in stem or 'sim_ood' in stem:
                combo = 'sim+ood'
            else:
                continue

            # Extract thresholds
            m_sim = re.search(r'sim(-?[\d.]+)', stem)
            m_deg = re.search(r'deg(-?[\d.]+)', stem)
            m_ood = re.search(r'ood(-?[\d.]+)', stem)
            sim_v = m_sim.group(1) if m_sim else '?'
            deg_v = m_deg.group(1) if m_deg else '?'
            ood_v = m_ood.group(1) if m_ood else '?'

            # Extract ptb
            m_ptb = re.search(r'ptb(\d+\.\d+)', stem)
            ptb = float(m_ptb.group(1)) if m_ptb else None
            if ptb is None:
                continue

            # Use a unique key that includes thresholds to distinguish variants
            label = f"{combo} (s={sim_v}, d={deg_v}, o={ood_v})"

            seeds = parse_seeds_from_log(fpath)
            if seeds:
                mean, std, n, vals_str = seeds_to_stat(seeds)
                if label not in data:
                    data[label] = {}
                data[label][ptb] = {"seeds": seeds, "mean": mean, "std": std, "n": n, "seeds_str": vals_str}

    return data


# ============================================================
# EDGE BINARY CLASSIFICATION METRICS
# ============================================================

def collect_all_edge_metrics():
    """
    Parse combo_filters_midmetrics logs WITH combo context.

    Returns:
      edge_src[combo][source][ptb] = list of {tpr, tnr, auc, f1}
      tip_src[combo][tip][ptb]     = list of {tpr, tnr, auc, f1, precision}
      combo_configs[combo]         = {sim, deg, ood, active_tips}
    """
    edge_src = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    tip_src  = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    combo_configs = {}

    d = LOGS_DIR / "combo_filters_midmetrics"
    if not d.exists():
        return edge_src, tip_src, combo_configs

    for fname in sorted(os.listdir(d)):
        if not fname.endswith('.log'):
            continue

        with open(d / fname, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        stem = fname.replace('.log', '')
        m_ptb = re.search(r'ptb(\d+\.\d+)', stem)
        ptb = float(m_ptb.group(1)) if m_ptb else None
        if ptb is None or ptb == 0.0:
            continue

        # --- Determine combo ---
        if 'all3' in stem:   combo = 'all3'
        elif 'deg_ood' in stem: combo = 'deg+ood'
        elif 'sim_deg' in stem: combo = 'sim+deg'
        elif 'sim_ood' in stem: combo = 'sim+ood'
        else: continue

        # --- Thresholds ---
        m_sim = re.search(r'sim(-?[\d.]+)', stem)
        m_deg = re.search(r'deg(-?[\d.]+)', stem)
        m_ood = re.search(r'ood(-?[\d.]+)', stem)
        sim_v = float(m_sim.group(1)) if m_sim else None
        deg_v = float(m_deg.group(1)) if m_deg else None
        ood_v = float(m_ood.group(1)) if m_ood else None

        active = []
        if sim_v is not None and sim_v != -1.0: active.append('sim_pt')
        if deg_v is not None and deg_v != -1.0: active.append('degree_pt')
        if ood_v is not None and ood_v != -1.0: active.append('out_detect_pt')

        if combo not in combo_configs:
            combo_configs[combo] = {'sim': sim_v, 'deg': deg_v, 'ood': ood_v, 'active': active}

        # --- Parse lines ---
        lines = content.split('\n')
        current_n_neg = None

        for line in lines:
            if 'Edge Detection |' in line and 'Detail' not in line:
                d_line = _parse_kv_line(line)
                source = d_line.get('source', 'unknown')
                tp = d_line.get('tp', 0) or 0
                fp = d_line.get('fp', 0) or 0
                fn = d_line.get('fn', 0) or 0
                tn = d_line.get('tn', 0) or 0

                tpr = tp / (tp + fn) if (tp + fn) > 0 else None
                tnr = tn / (tn + fp) if (tn + fp) > 0 else None
                current_n_neg = tn + fp

                edge_src[combo][source][ptb].append({
                    'tpr': tpr, 'tnr': tnr,
                    'auc': d_line.get('auc'), 'f1': d_line.get('f1'),
                })

            elif 'Tip Detection |' in line:
                d_line = _parse_kv_line(line)
                tip = d_line.get('tip', 'unknown')
                edge_tp = d_line.get('edge_tp', 0) or 0
                edge_fp = d_line.get('edge_fp', 0) or 0
                edge_fn = d_line.get('edge_fn', 0) or 0

                tpr_tip = edge_tp / (edge_tp + edge_fn) if (edge_tp + edge_fn) > 0 else None
                tnr_tip = None
                if current_n_neg is not None and current_n_neg > 0:
                    edge_tn = current_n_neg - edge_fp
                    tnr_tip = edge_tn / current_n_neg if edge_tn >= 0 else 0.0

                tip_src[combo][tip][ptb].append({
                    'tpr': tpr_tip,
                    'tnr': tnr_tip,
                    'auc': d_line.get('edge_auc'),
                    'f1': d_line.get('edge_f1'),
                    'precision': d_line.get('edge_precision'),
                })

    return edge_src, tip_src, combo_configs


def _parse_kv_line(line):
    """Parse a | key=value | key=value | ... line into a dict."""
    d = {}
    for part in line.split('|')[1:]:
        part = part.strip()
        if '=' in part:
            k, v = part.split('=', 1)
            k = k.strip(); v = v.strip()
            try:
                d[k] = None if v == 'nan' else (float(v) if '.' in v else int(v))
            except ValueError:
                d[k] = v
    return d


def avg_edge(entries, key):
    vals = [e.get(key) for e in entries if e.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


# ============================================================
# PART TWO REPORT GENERATOR
# ============================================================

OUTPUT_PART2 = BASE_DIR / "reports" / "meeting15_part2.html"

def generate_part2(edge_src, tip_src, combo_configs):
    """Generate Part Two: edge anomaly detection, organized by combo context."""

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    PTBS = [0.05, 0.10, 0.15, 0.20, 0.25]
    COMBO_ORDER = ["sim+deg", "sim+ood", "deg+ood", "all3"]  # 2-filter then 3-filter

    def cell_from_entries(entries, key):
        vals = [e.get(key) for e in entries if e.get(key) is not None]
        if not vals: return None
        mean = sum(vals) / len(vals)
        std = (sum((v - mean)**2 for v in vals) / len(vals))**0.5 if len(vals) > 1 else 0.0
        return (mean, std, len(vals))

    def fmt_cell(cell):
        if cell is None: return '<td class="na">—</td>'
        m, s, n = cell
        if s < 0.0001: return f'<td>{m:.4f}</td>'
        return f'<td>{m:.4f}&plusmn;{s:.4f}</td>'

    # ================================================================
    # Build: one master table per metric
    # Rows are organized as [combo] -> [source]
    # ================================================================
    def build_combo_aware_table(edge_dict, tip_dict, metric_key):
        """Rows: combo header + its edge sources + its active tips."""
        rows = []
        for combo in COMBO_ORDER:
            if combo not in combo_configs:
                continue
            active_tips = combo_configs[combo]['active']

            # Combo header row
            rows.append(('combo_header', combo, [combo] + [None]*len(PTBS)))

            # Edge Detection sources for this combo
            for src in ['tau_tune', 'filter_module', 'out_detect_pt_edges']:
                if src in edge_dict.get(combo, {}):
                    row = [f'  {src}']
                    for ptb in PTBS:
                        entries = edge_dict[combo][src].get(ptb, [])
                        row.append(cell_from_entries(entries, metric_key))
                    rows.append(('edge_src', src, row))

            # Active Tip sources for this combo
            for tip in ['sim_pt', 'degree_pt', 'out_detect_pt']:
                if tip in active_tips and tip in tip_dict.get(combo, {}):
                    row = [f'  {tip} (tip)']
                    for ptb in PTBS:
                        entries = tip_dict[combo][tip].get(ptb, [])
                        row.append(cell_from_entries(entries, metric_key))
                    rows.append(('tip_src', tip, row))
                elif tip not in active_tips:
                    # Show as inactive
                    row = [f'  {tip} (inactive)']
                    for ptb in PTBS:
                        row.append(None)
                    rows.append(('tip_inactive', tip, row))

        return rows

    # Build tables for all 6 metrics + compute derived FPR/FNR/balanced-AUC
    # Edge Detection sources have tpr, tnr, f1 directly; tips have tpr, tnr, f1
    # FPR = 1 - TNR,  FNR = 1 - TPR,  Balanced AUC = (TPR + TNR) / 2
    tables = {}
    for metric_key, label in [('tpr','TPR (Recall)'), ('tnr','TNR'), ('f1','F1')]:
        tables[label] = build_combo_aware_table(edge_src, tip_src, metric_key)

    # Derived metrics: FPR, FNR, Balanced AUC = (TPR+TNR)/2
    for derived_key, derived_label, compute_fn in [
        ('fpr', 'FPR (False Positive Rate)', lambda e: (1.0 - e['tnr']) if e.get('tnr') is not None else None),
        ('fnr', 'FNR (False Negative Rate)', lambda e: (1.0 - e['tpr']) if e.get('tpr') is not None else None),
        ('balanced_auc', 'Binary Classification AUC = (TPR+TNR)/2',
         lambda e: ((e['tpr'] + e['tnr']) / 2.0) if (e.get('tpr') is not None and e.get('tnr') is not None) else None),
    ]:
        # Create derived edge_src and tip_src with the computed metric
        derived_edge = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        derived_tip  = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for combo in edge_src:
            for src in edge_src[combo]:
                for ptb, entries in edge_src[combo][src].items():
                    for e in entries:
                        v = compute_fn(e)
                        if v is not None:
                            derived_edge[combo][src][ptb].append({derived_key: v})
        for combo in tip_src:
            for tip in tip_src[combo]:
                for ptb, entries in tip_src[combo][tip].items():
                    for e in entries:
                        v = compute_fn(e)
                        if v is not None:
                            derived_tip[combo][tip][ptb].append({derived_key: v})
        tables[derived_label] = build_combo_aware_table(derived_edge, derived_tip, derived_key)

    # ================================================================
    # Render HTML
    # ================================================================
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Meeting 15 — Part 2: Edge Anomaly Detection</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #fff; color: #1a1a1a; font-size: 12px; line-height:1.5; }}
  .container {{ max-width: 1200px; margin:0 auto; padding: 28px 16px; }}
  header {{ margin-bottom: 24px; border-bottom: 1.5px solid #2563eb; padding-bottom: 12px; }}
  header h1 {{ font-size: 18px; font-weight: 700; }}
  header .meta {{ font-size: 10px; color: #888; margin-top: 3px; }}
  h2 {{ font-size: 15px; font-weight: 600; margin: 32px 0 12px; color: #333; border-bottom: 1px solid #eee; padding-bottom: 6px; }}
  h3 {{ font-size: 13px; font-weight: 600; margin: 22px 0 8px; color: #444; }}
  table.data-table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin: 6px 0 16px; font-variant-numeric: tabular-nums; }}
  table.data-table thead th {{ background: #f5f5f5; font-weight: 600; text-align: center; padding: 5px 6px; border-bottom: 1.5px solid #ddd; white-space: nowrap; }}
  table.data-table tbody td {{ text-align: center; padding: 3px 6px; border-bottom: 1px solid #eee; }}
  table.data-table tbody tr:nth-child(even) {{ background: #fafafa; }}
  .combo-header {{ background: #e8f0fe; font-weight: 700; }}
  .combo-header td {{ text-align: left !important; padding: 6px 6px !important; border-bottom: 1.5px solid #bbb !important; }}
  .edge-src {{ }}
  .tip-src {{ }}
  .tip-inactive {{ color: #ccc; }}
  .na {{ color: #bbb; font-style: italic; }}
  .note {{ font-size: 10px; color: #999; margin: 2px 0 8px; }}
  .def-box {{ background: #f8fafc; border: 1px solid #e0e0e0; border-radius: 6px; padding: 14px 18px; margin: 10px 0 18px; font-size: 11px; line-height: 1.7; }}
  .def-box table {{ border-collapse: collapse; margin: 6px 0; }}
  .def-box table td {{ padding: 2px 12px 2px 0; font-size: 11px; }}
  .formula {{ font-family: 'SF Mono', 'Consolas', monospace; background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }}
  .config-table td {{ padding: 3px 10px; font-size: 11px; }}
  hr {{ border: none; border-top: 1px solid #eee; margin: 24px 0; }}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>Meeting 15 — Part 2: Intermediate Filter Edge Anomaly Detection</h1>
  <div class="meta">
    Generated: {now} &nbsp;|&nbsp; Cora / Meta_Self / 5-shot split-1 &nbsp;|&nbsp;
    Peak BB: permE-maskN, lr=0.001, r=0.3
  </div>
</header>

<h2>Definitions</h2>
<div class="def-box">
  <p><strong>Task:</strong> binary classification of edges — <strong>Positive = Anomaly</strong> (attack edge, added/deleted by Metattack) vs <strong>Negative = Normal</strong> (clean original edge).</p>
  <table>
    <tr><td><strong>TP</strong> = attack edges correctly flagged</td><td><strong>FP</strong> = clean edges incorrectly flagged</td></tr>
    <tr><td><strong>TN</strong> = clean edges correctly left alone</td><td><strong>FN</strong> = attack edges missed</td></tr>
  </table>
  <p style="margin-top:8px;">
    <span class="formula">TPR (Recall) = TP / (TP + FN)</span> — attack edge detection rate.<br>
    <span class="formula">TNR = TN / (TN + FP)</span> — clean edge preservation rate.<br>
    <span class="formula">FPR = FP / (FP + TN) = 1 &minus; TNR</span> — false alarm rate. Should be low.<br>
    <span class="formula">FNR = FN / (FN + TP) = 1 &minus; TPR</span> — miss rate. Should be low.<br>
    <span class="formula">F1 = 2 &times; Prec &times; Recall / (Prec + Recall)</span>, where <span class="formula">Prec = TP / (TP + FP)</span>.<br>
    <span class="formula">Binary Classification AUC = (TPR + TNR) / 2</span> — balanced accuracy of the single operating point. Ranges [0, 1], 0.5 = random.
  </p>
  <p style="margin-top:8px; color:#888;">
    Tip TNR is estimated: <span class="formula">(N<sub>clean</sub> &minus; FP<sub>tip</sub>) / N<sub>clean</sub></span>
    where N<sub>clean</sub> comes from the paired Edge Detection line. Inactive tips are grayed out.
  </p>
</div>

<h2>1. Combo Configuration Overview</h2>
<table class="data-table config-table">
<thead><tr><th>Combo</th><th># Active</th><th>sim_pt</th><th>degree_pt</th><th>out_detect_pt</th></tr></thead>
<tbody>
'''

    for combo in COMBO_ORDER:
        if combo not in combo_configs: continue
        cfg = combo_configs[combo]
        active = cfg['active']
        html += '<tr>'
        html += f'<td style="text-align:left;font-weight:600;">{combo}</td>'
        html += f'<td>{len(active)}</td>'
        for tip in ['sim_pt', 'degree_pt', 'out_detect_pt']:
            val = cfg.get({'sim_pt':'sim','degree_pt':'deg','out_detect_pt':'ood'}[tip], '—')
            if tip in active:
                html += f'<td style="color:#059669;font-weight:600;">{val} (ON)</td>'
            else:
                html += f'<td style="color:#ccc;">{val} (off)</td>'
        html += '</tr>\n'

    html += '''
</tbody></table>
<p class="note">Active thresholds: sim=0.3, deg=1, ood=0.5. Inactive tips are set to -1.0 (never trigger). Progression: 2-filter combos → 3-filter combo (all3).</p>

<h2>2. Edge Anomaly Detection by Combo Context</h2>
<p class="note">Each table below shows per-combo, per-source metrics. Rows are grouped by combo (blue header) with edge detection sources and active tip sources nested underneath. Inactive tips are grayed out.</p>
'''

    # Render 6 metric tables
    labels = list(tables.keys())
    for i, metric_label in enumerate(labels):
        rows = tables[metric_label]
        html += f'<h3>2.{chr(65+i)} {metric_label}</h3>\n'
        html += '<table class="data-table">\n<thead>\n<tr>\n<th>Combo / Source</th>\n'
        for ptb in PTBS:
            html += f'<th>{ptb:.2f}</th>'
        html += '\n</tr>\n</thead>\n<tbody>\n'

        for row_type, src_name, row_data in rows:
            css_class = row_type.replace('_', '-')
            html += f'<tr class="{css_class}">'
            for i, cell in enumerate(row_data):
                if i == 0:
                    html += f'<td>{cell}</td>'
                else:
                    html += fmt_cell(cell)
            html += '</tr>\n'

        html += '</tbody>\n</table>\n'

    html += '''
</div>
</body>
</html>'''
    return html


# ============================================================
# HTML GENERATION (Part One)
# ============================================================

def render_table(headers, rows, precision=4):
    """Minimal HTML table."""
    html = '<table class="data-table">\n<thead>\n<tr>\n'
    for h in headers:
        html += f'<th>{h}</th>'
    html += '\n</tr>\n</thead>\n<tbody>\n'
    for i, row in enumerate(rows):
        html += '<tr>\n'
        for cell in row:
            if cell is None:
                html += '<td class="na">—</td>'
            elif isinstance(cell, float):
                html += f'<td>{cell:.{precision}f}</td>'
            else:
                html += f'<td>{cell}</td>'
        html += '</tr>\n'
    html += '</tbody>\n</table>\n'
    return html


def generate_report():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    print("Parsing single filter logs...")
    single = collect_single_filter()
    print("Parsing GPPT logs...")
    gppt = collect_gppt()
    print("Parsing GCL logs (incl. clean from eval_pretrain)...")
    gcl = collect_gcl()
    print("Parsing combo filter logs...")
    combos = collect_combos()

    # ================================================================
    # Build master table rows
    # ================================================================
    master_rows = []  # list of (section_type, label, [values_per_ptb])
    seed_notes = []   # per-model seed detail strings

    # --- Single filters ---
    filter_order = [("sim", "sim_pt"), ("degree", "degree_pt"), ("ood", "out_detect_pt")]
    for ftype, flabel in filter_order:
        for thr in sorted(single.get(ftype, {}).keys()):
            label = f"{flabel}={thr}"
            row = []
            seed_parts = []
            for ptb in PTB_LEVELS:
                entry = single[ftype][thr].get(ptb)
                if entry and entry["n"] > 0:
                    row.append((entry["mean"], entry["std"]))
                    seed_parts.append(f"ptb={ptb:.2f}: [{entry['seeds_str']}]")
                else:
                    row.append(None)
            master_rows.append(("single", label, row))
            if seed_parts:
                seed_notes.append((label, "; ".join(seed_parts)))

    # --- Combos ---
    for label in sorted(combos.keys()):
        row = []
        seed_parts = []
        for ptb in PTB_LEVELS:
            entry = combos[label].get(ptb)
            if entry and entry["n"] > 0:
                row.append((entry["mean"], entry["std"]))
                seed_parts.append(f"ptb={ptb:.2f}: [{entry['seeds_str']}]")
            else:
                row.append(None)
        master_rows.append(("combo", label, row))
        if seed_parts:
            seed_notes.append((label, "; ".join(seed_parts)))

    # --- GPPT ---
    gppt_row = []
    gppt_seeds = []
    for ptb in PTB_LEVELS:
        entry = gppt.get(ptb)
        if entry and entry["n"] > 0:
            gppt_row.append((entry["mean"], entry["std"]))
            gppt_seeds.append(f"ptb={ptb:.2f}: [{entry['seeds_str']}]")
        else:
            gppt_row.append(None)
    master_rows.append(("baseline", "GPPT", gppt_row))
    if gppt_seeds:
        seed_notes.append(("GPPT", "; ".join(gppt_seeds)))

    # --- GCL ---
    gcl_row = []
    gcl_seeds = []
    for ptb in PTB_LEVELS:
        entry = gcl.get(ptb)
        if entry and entry["n"] > 0:
            gcl_row.append((entry["mean"], entry["std"]))
            gcl_seeds.append(f"ptb={ptb:.2f}: [{entry['seeds_str']}]")
        else:
            gcl_row.append(None)
    master_rows.append(("baseline", "GCL (Linear Probe)", gcl_row))
    if gcl_seeds:
        seed_notes.append(("GCL", "; ".join(gcl_seeds)))

    # ================================================================
    # Render HTML
    # ================================================================
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Meeting 15 — Summary</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #fff; color: #1a1a1a; font-size: 12px; line-height:1.5; }}
  .container {{ max-width: 1300px; margin:0 auto; padding: 28px 16px; }}
  header {{ margin-bottom: 24px; border-bottom: 1.5px solid #2563eb; padding-bottom: 12px; }}
  header h1 {{ font-size: 18px; font-weight: 700; }}
  header .meta {{ font-size: 10px; color: #888; margin-top: 3px; }}
  h2 {{ font-size: 14px; font-weight: 600; margin: 28px 0 10px; color: #333; }}
  h3 {{ font-size: 13px; font-weight: 600; margin: 20px 0 8px; color: #444; }}
  table.data-table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin: 6px 0 14px; font-variant-numeric: tabular-nums; }}
  table.data-table thead th {{ background: #f5f5f5; font-weight: 600; text-align: center; padding: 4px 6px; border-bottom: 1.5px solid #ddd; white-space: nowrap; }}
  table.data-table tbody td {{ text-align: center; padding: 3px 6px; border-bottom: 1px solid #eee; }}
  table.data-table tbody tr:nth-child(even) {{ background: #fafafa; }}
  .section-single {{ }}
  .section-combo {{ background: #f0f4ff; }}
  .section-baseline {{ background: #fef9e7; font-weight: 600; }}
  .na {{ color: #bbb; font-style: italic; }}
  .note {{ font-size: 10px; color: #999; margin-top: 2px; }}
  .seed-detail {{ font-size: 10px; color: #777; font-family: 'SF Mono', 'Consolas', monospace; margin: 2px 0; }}
  details {{ margin: 6px 0 12px; }}
  details summary {{ font-size: 11px; color: #555; cursor: pointer; }}
  hr {{ border: none; border-top: 1px solid #eee; margin: 20px 0; }}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>Meeting 15 &mdash; RobustPrompt-T Summary (Peak Backbone Only)</h1>
  <div class="meta">
    Generated: {now} &nbsp;|&nbsp; Cora / Meta_Self / 5-shot split-1 &nbsp;|&nbsp;
    Peak BB: permE-maskN, lr=0.001, r=0.3 &nbsp;|&nbsp;
    Accuracy = mean &plusmn; std across all 5 seeds
  </div>
</header>

<h2>1. Accuracy Across All Models and Perturbation Levels</h2>
<p class="note">
  Each cell: <strong>mean &plusmn; std</strong> (all 5 seeds included, no min-removal).
  <span style="background:#f0f4ff">&nbsp;Blue&nbsp;</span> = combo,
  <span style="background:#fef9e7">&nbsp;Yellow&nbsp;</span> = baseline.
  <code>—</code> = no data.
</p>
<table class="data-table">
<thead><tr><th>Model</th>'''

    for p in PTB_LEVELS:
        html += f'<th>{p:.2f}</th>'
    html += '</tr></thead><tbody>'

    for section, label, row in master_rows:
        css = f'section-{section}'
        html += f'<tr class="{css}"><td>{label}</td>'
        for cell in row:
            if cell is None:
                html += '<td class="na">—</td>'
            else:
                mean, std = cell
                if std == 0 or std < 0.0001:
                    html += f'<td>{mean:.4f}</td>'
                else:
                    html += f'<td>{mean:.4f}&plusmn;{std:.4f}</td>'
        html += '</tr>\n'
    html += '</tbody></table>\n'

    # Seed details (collapsible)
    html += '<details>\n<summary>Per-model seed details (click to expand)</summary>\n'
    for label, detail in seed_notes:
        html += f'<div class="seed-detail"><strong>{label}:</strong> {detail}</div>\n'
    html += '</details>\n'

    html += '''
</div>
</body>
</html>'''

    return html


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Meeting 15 Report Generator v3")
    print("=" * 60)

    # Part One
    print("\n--- Part One: Accuracy Report ---")
    html1 = generate_report()
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html1)
    print(f"  -> {OUTPUT_HTML} ({OUTPUT_HTML.stat().st_size / 1024:.1f} KB)")

    # Part Two: Edge Anomaly Detection
    print("\n--- Part Two: Edge Anomaly Detection ---")
    print("Collecting edge metrics (with combo context)...")
    edge_src, tip_src, combo_configs = collect_all_edge_metrics()

    # Count records
    for label, data in [("Edge Detection sources", edge_src), ("Tip Detection sources", tip_src)]:
        total = sum(len(entries) for combo_data in data.values() for src_data in combo_data.values() for entries in src_data.values())
        print(f"  {label}: {len(data)} combos, {total} records")

    print(f"  Combo configs: {list(combo_configs.keys())}")
    for combo, cfg in combo_configs.items():
        print(f"    {combo}: active={cfg['active']}")

    html2 = generate_part2(edge_src, tip_src, combo_configs)
    with open(OUTPUT_PART2, 'w', encoding='utf-8') as f:
        f.write(html2)
    print(f"  -> {OUTPUT_PART2} ({OUTPUT_PART2.stat().st_size / 1024:.1f} KB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
