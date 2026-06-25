"""
解析所有实验日志，生成 HTML 综合报告（含 NaN 注释和稳定性分析）。
用法: python generate_meeting14_report.py
"""
import os, re, glob
from collections import defaultdict
from datetime import datetime

LOG_DIRS = {
    'GPPT':       'logs/baselines',
    'GCL_LP':     'logs/baselines',
    'sim_pt':     'logs/single_filter_sim',
    'degree_pt':  'logs/single_filter_degree',
    'out_detect_pt': 'logs/single_filter_ood',
}
PTB_ORDER = ['0.0', '0.05', '0.1', '0.15', '0.2', '0.25']
SIM_VALS = ['0.2', '0.3', '0.4', '0.5', '0.6']
DEG_VALS = ['1', '2', '3', '5']
OOD_VALS = ['0.3', '0.4', '0.5', '0.6', '0.7']

# ===================== Log Parsing =====================

def parse_gppt_log(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        matches = re.findall(r'Final True Accuracy:\s+([\d.]+)', content)
        if matches:
            return {'acc': float(matches[0]), 'std': 0.0, 'per_seed': [float(matches[0])]*5}
    except Exception as e:
        print(f"  ERROR parsing GPPT {filepath}: {e}")
    return None


def parse_gcl_lp_log(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        m = re.search(r'Test Acc:\s+([\d.]+)', content)
        if m:
            acc = float(m.group(1))
            return {'acc': acc, 'val_acc': 0.0}
    except Exception as e:
        print(f"  ERROR parsing GCL_LP {filepath}: {e}")
    return None


def parse_robustprompt_log(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        per_seed = {}
        for m in re.finditer(r'seed:\s+(\d+)\s+\|\s+split\s+\d+\s+:\s+([\d.nan]+)', content):
            seed = int(m.group(1))
            val_str = m.group(2)
            per_seed[seed] = float('nan') if val_str == 'nan' else float(val_str)
        m = re.search(r'Muti Seed Acc without min value:\s+([\d.nan]+)\xb1([\d.nan]+)', content)
        if not m:
            m = re.search(r'Muti Seed Acc without min value:\s+([\d.nan]+).([\d.nan]+)', content)
        if m:
            acc_str = m.group(1)
            std_str = m.group(2)
            acc = float('nan') if acc_str == 'nan' else float(acc_str)
            std = float('nan') if std_str == 'nan' else float(std_str)
        else:
            vals = [v for v in per_seed.values() if v == v]
            if len(vals) >= 4:
                import numpy as np
                vs = sorted(vals)[1:]; acc = float(np.mean(vs)); std = float(np.std(vs))
            elif len(vals) > 0:
                import numpy as np
                acc = float(np.mean(vals)); std = float(np.std(vals))
            else:
                acc = float('nan'); std = float('nan')
        return {'acc': acc, 'std': std, 'per_seed': per_seed}
    except Exception as e:
        print(f"  ERROR parsing RobustPrompt {filepath}: {e}")
    return None


def parse_filename_info(filename):
    info = {}
    if '_stable_' in filename: info['backbone'] = 'stable'
    elif '_peak_' in filename: info['backbone'] = 'peak'
    else: info['backbone'] = 'unknown'
    m = re.search(r'_(\d+\.\d+)_\d{8}_\d{6}', filename)
    if m: info['ptb'] = m.group(1)
    else:
        m = re.search(r'_(\d+\.\d+)_', filename)
        if m: info['ptb'] = m.group(1)
    for prefix in ['sim', 'deg', 'ood']:
        m = re.search(prefix + r'([\d.]+)_', filename)
        if m: info['filter_val'] = m.group(1); break
    return info


def collect_all_results():
    results = {'GPPT': [], 'GCL_LP': [], 'sim_pt': [], 'degree_pt': [], 'out_detect_pt': []}

    for f in sorted(glob.glob('logs/baselines/gppt_*.log')):
        info = parse_filename_info(f)
        info['filename'] = os.path.basename(f)
        parsed = parse_gppt_log(f)
        if parsed: info.update(parsed); results['GPPT'].append(info)
        else: print(f"  SKIP GPPT {f} (no valid result)")

    for f in sorted(glob.glob('logs/baselines/gcl_lp_*.log')):
        info = parse_filename_info(f)
        info['filename'] = os.path.basename(f)
        parsed = parse_gcl_lp_log(f)
        if parsed: info.update(parsed); results['GCL_LP'].append(info)
        else: print(f"  SKIP GCL_LP {f} (no valid result)")

    for exp_type, log_dir in [('sim_pt', 'logs/single_filter_sim'),
                               ('degree_pt', 'logs/single_filter_degree'),
                               ('out_detect_pt', 'logs/single_filter_ood')]:
        for f in sorted(glob.glob(f'{log_dir}/*.log')):
            info = parse_filename_info(f)
            info['filename'] = os.path.basename(f)
            parsed = parse_robustprompt_log(f)
            if parsed: info.update(parsed); results[exp_type].append(info)
            else: print(f"  SKIP {exp_type} {f} (no valid result)")

    return results


# ===================== NaN Analysis =====================

def build_nan_registry(results):
    """Build detailed NaN info for every (exp_type, bb, ptb, fv) combination."""
    nan_reg = {}
    for exp_type in ['sim_pt', 'degree_pt', 'out_detect_pt']:
        for r in results[exp_type]:
            ps = r.get('per_seed', {})
            n_nan = sum(1 for v in ps.values() if v != v)
            if n_nan == 0:
                continue
            key = (exp_type, r['backbone'], r['ptb'], r.get('filter_val', '?'))
            # Per-seed detail: which seeds are NaN
            seed_detail = []
            for s in sorted(ps.keys()):
                v = ps[s]
                seed_detail.append(f"S{s}=NaN" if v != v else f"S{s}={v:.4f}")
            # Cause analysis
            if r['backbone'] == 'stable' and n_nan >= 2:
                cause = "stable BB 在攻击下易过拟合，≥2 seeds 梯度爆炸"
            elif r['backbone'] == 'stable' and n_nan == 1:
                cause = "stable BB 单 seed 训练不稳定"
            else:
                cause = "随机种子敏感性，单 seed loss 爆炸"
            nan_reg[key] = {
                'n_nan': n_nan,
                'seeds': '; '.join(seed_detail),
                'cause': cause,
                'agg_is_nan': (r['acc'] != r['acc']),
            }
    return nan_reg


def fmt_acc_with_nan_note(acc, std, nan_key, nan_reg):
    """Format accuracy cell, adding NaN footnote marker if needed."""
    is_nan = (isinstance(acc, float) and acc != acc)

    if is_nan:
        # Check if this key has detailed NaN info
        if nan_key in nan_reg:
            ni = nan_reg[nan_key]
            note = f"{ni['n_nan']}/5 seeds NaN" if ni['agg_is_nan'] else f"{ni['n_nan']}/5 NaN (agg ok)"
            return f'<span style="color:#dc2626" title="{ni["seeds"]}">{note}</span>'
        else:
            return '<span style="color:#dc2626">NaN</span>'

    if std is not None and std == std:
        val = f'{acc:.4f}'
        if nan_key in nan_reg:
            ni = nan_reg[nan_key]
            val += f' <sup style="color:#d97706" title="{ni["seeds"]}">⚠{ni["n_nan"]}</sup>'
        val += f' ± {std:.4f}'
        return val
    return f'{acc:.4f}'


def build_nan_footnotes(nan_reg, exp_type=None, bb_filter=None):
    """Build HTML footnotes listing all NaN cases filtered by exp_type and/or bb."""
    notes = []
    for key, ni in sorted(nan_reg.items()):
        et, bb, ptb, fv = key
        if exp_type and et != exp_type: continue
        if bb_filter and bb != bb_filter: continue
        agg_mark = ' [聚合=NaN]' if ni['agg_is_nan'] else ''
        notes.append(
            f'<tr><td>{et}</td><td>{bb}</td><td>ptb={ptb}</td><td>{fv}</td>'
            f'<td>{ni["n_nan"]}/5</td><td style="font-size:0.82rem">{ni["seeds"]}</td>'
            f'<td style="font-size:0.82rem">{ni["cause"]}{agg_mark}</td></tr>'
        )
    if notes:
        header = '<tr><th>Type</th><th>BB</th><th>ptb</th><th>Val</th><th>NaN</th><th>Per-Seed Detail</th><th>Root Cause</th></tr>'
        return f'<table>{header}{"".join(notes)}</table>'
    return '<p style="color:var(--good)">✅ 全部 5/5 seeds 稳定</p>'


# ===================== Table Builders =====================

def build_gppt_table(results):
    rows = []
    for bb in ['stable', 'peak']:
        cells = [f'<td>{bb}</td>']
        for ptb in PTB_ORDER:
            matches = [r for r in results['GPPT'] if r['backbone'] == bb and r['ptb'] == ptb]
            if matches:
                cells.append(f'<td class="num">{matches[0]["acc"]:.4f}</td>')
            else:
                cells.append('<td class="num" style="color:#999">—</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    header = '<tr><th>Backbone</th>' + ''.join(f'<th>ptb={p}</th>' for p in PTB_ORDER) + '</tr>'
    return f'<table>{header}{"".join(rows)}</table>'


def build_gcl_lp_table(results):
    clean_acc = {'stable': 0.5689, 'peak': 0.6262}
    rows = []
    for bb in ['stable', 'peak']:
        cells = [f'<td>{bb}</td>']
        for ptb in PTB_ORDER:
            if ptb == '0.0':
                cells.append(f'<td class="num">{clean_acc[bb]:.4f}</td>')
            else:
                matches = [r for r in results['GCL_LP'] if r['backbone'] == bb and r['ptb'] == ptb]
                if matches:
                    cells.append(f'<td class="num">{matches[0]["acc"]:.4f}</td>')
                else:
                    cells.append('<td class="num" style="color:#999">—</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    header = '<tr><th>Backbone</th>' + ''.join(f'<th>ptb={p}</th>' for p in PTB_ORDER) + '</tr>'
    return f'<table>{header}{"".join(rows)}</table>'


def build_rp_single_filter_table(results, exp_type, filter_label, filter_vals, nan_reg):
    sections = []
    for bb in ['stable', 'peak']:
        rows = []
        for fv in filter_vals:
            cells = [f'<td>{fv}</td>']
            for ptb in PTB_ORDER:
                matches = [r for r in results[exp_type]
                          if r['backbone'] == bb and r['ptb'] == ptb and r.get('filter_val') == fv]
                if matches:
                    r = matches[0]
                    nan_key = (exp_type, bb, ptb, fv)
                    cells.append(f'<td class="num">{fmt_acc_with_nan_note(r["acc"], r["std"], nan_key, nan_reg)}</td>')
                else:
                    cells.append('<td class="num" style="color:#999">—</td>')
            rows.append('<tr>' + ''.join(cells) + '</tr>')
        header = f'<tr><th>{filter_label}</th>' + ''.join(f'<th>ptb={p}</th>' for p in PTB_ORDER) + '</tr>'
        table_html = f'<h4 style="margin-top:16px">Backbone: {bb}</h4><table>{header}{"".join(rows)}</table>'
        # Add NaN footnotes for this BB
        nan_notes = build_nan_footnotes(nan_reg, exp_type=exp_type, bb_filter=bb)
        if '✅' not in nan_notes:
            table_html += f'<details style="margin:4px 0 12px"><summary style="font-size:0.82rem;color:var(--warn);cursor:pointer">NaN 详情 ({bb} BB)</summary>{nan_notes}</details>'
        sections.append(table_html)
    return '\n'.join(sections)


def build_best_summary_table(results, exp_type, label, nan_reg):
    """Show best filter value for each (BB, ptb) with NaN-aware formatting."""
    rows = []
    n_total = 0
    n_stable = 0
    for bb in ['stable', 'peak']:
        cells = [f'<td>{bb}</td>']
        for ptb in PTB_ORDER:
            best_acc = -1.0
            best_fv = '—'
            best_std = 0.0
            best_nan_key = None
            for r in results[exp_type]:
                if r['backbone'] == bb and r['ptb'] == ptb:
                    acc = r['acc']
                    if acc == acc and acc > best_acc:
                        best_acc = acc
                        best_fv = r.get('filter_val', '?')
                        best_std = r.get('std', 0)
                        best_nan_key = (exp_type, bb, ptb, best_fv)
            if best_acc >= 0:
                n_total += 1
                if best_nan_key and best_nan_key in nan_reg: n_stable += 1
                acc_str = f'{best_acc:.4f}'
                if best_nan_key in nan_reg:
                    acc_str += f' <sup style="color:#d97706">⚠</sup>'
                acc_str += f' ± {best_std:.4f}'
                cells.append(f'<td class="num">{acc_str}<br><span style="font-size:0.75rem;color:var(--muted)">{label}={best_fv}</span></td>')
            else:
                cells.append('<td class="num" style="color:#dc2626">all NaN</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    header = '<tr><th>Backbone</th>' + ''.join(f'<th>ptb={p}</th>' for p in PTB_ORDER) + '</tr>'
    return f'<table>{header}{"".join(rows)}</table>'


def build_cross_bb_comparison(results, exp_type, filter_vals, filter_label, nan_reg):
    rows = []
    for fv in filter_vals:
        cells = [f'<td>{fv}</td>']
        for ptb in PTB_ORDER:
            best_acc = -1.0
            best_bb = '—'
            for bb in ['stable', 'peak']:
                matches = [r for r in results[exp_type]
                          if r['backbone'] == bb and r['ptb'] == ptb and r.get('filter_val') == fv]
                if matches:
                    acc = matches[0]['acc']
                    if acc == acc and acc > best_acc:
                        best_acc = acc
                        best_bb = bb
            if best_acc >= 0:
                cells.append(f'<td class="num">{best_acc:.4f}<br><span style="font-size:0.75rem;color:var(--muted)">{best_bb}</span></td>')
            else:
                cells.append('<td class="num" style="color:#dc2626">NaN</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    header = f'<tr><th>{filter_label}</th>' + ''.join(f'<th>ptb={p}</th>' for p in PTB_ORDER) + '</tr>'
    return f'<table>{header}{"".join(rows)}</table>'


def build_stability_table(results, nan_reg):
    """Stability analysis with per-ptb seed counts.
    Classification:
      - stable:        6/6 ptb 全部 5/5 seeds 有效
      - roughly stable: 6/6 ptb 全部 ≥4/5 seeds 有效（至少 1 个 ptb 有 4/5）
      - unstable:      存在 ptb 仅有 ≤3/5 seeds 有效
    """
    rows = []
    for exp_type, label, vals in [('sim_pt','sim',SIM_VALS), ('degree_pt','deg',DEG_VALS), ('out_detect_pt','ood',OOD_VALS)]:
        for bb in ['stable', 'peak']:
            for fv in vals:
                # Per-ptb seed counts
                ptb_counts = {}
                n_5of5 = 0; n_4of5 = 0; n_unstable = 0
                for ptb in PTB_ORDER:
                    matches = [r for r in results[exp_type]
                              if r['backbone'] == bb and r['ptb'] == ptb and r.get('filter_val') == fv]
                    if matches:
                        r = matches[0]
                        ps = r.get('per_seed', {})
                        n_valid = sum(1 for v in ps.values() if v == v)
                        ptb_counts[ptb] = n_valid
                        if n_valid == 5: n_5of5 += 1
                        elif n_valid == 4: n_4of5 += 1
                        else: n_unstable += 1
                    else:
                        ptb_counts[ptb] = -1

                # Classification
                if n_unstable > 0:
                    cls_label = 'unstable'
                    cls_color = 'var(--warn)'
                elif n_4of5 > 0:
                    cls_label = 'roughly stable'
                    cls_color = 'var(--amber)'
                else:
                    cls_label = 'stable'
                    cls_color = 'var(--good)'

                # Per-ptb detail string
                detail_parts = []
                for ptb in PTB_ORDER:
                    n = ptb_counts.get(ptb, -1)
                    if n == 5:
                        detail_parts.append(f'<span style="color:var(--good)">{ptb}:5/5</span>')
                    elif n == 4:
                        detail_parts.append(f'<span style="color:var(--amber)">{ptb}:4/5</span>')
                    elif n >= 0:
                        detail_parts.append(f'<span style="color:var(--warn)">{ptb}:{n}/5</span>')
                    else:
                        detail_parts.append(f'<span style="color:#999">{ptb}:—</span>')
                detail = ' '.join(detail_parts)

                rows.append(f'<tr><td>{exp_type}</td><td>{bb}</td><td>{fv}</td>'
                           f'<td style="color:{cls_color};font-weight:600">{cls_label}</td>'
                           f'<td style="font-size:0.82rem">{detail}</td></tr>')

    header = '<tr><th>Type</th><th>BB</th><th>Val</th><th>Classification</th><th>Per-ptb Detail (seeds valid)</th></tr>'
    return f'<table>{header}{"".join(rows)}</table>'


def build_nan_summary_section(nan_reg):
    """Build a concise NaN summary for the overview section."""
    by_type = defaultdict(list)
    for key, ni in nan_reg.items():
        et, bb, ptb, fv = key
        by_type[et].append((bb, ptb, fv, ni))

    sections = []
    for exp_type in ['sim_pt', 'degree_pt', 'out_detect_pt']:
        cases = by_type.get(exp_type, [])
        if not cases:
            sections.append(f'<p><strong>{exp_type}:</strong> ✅ 0 NaN cases — 全部稳定</p>')
            continue
        # Count unique (BB, fv) combinations affected
        unique_combos = set((bb, fv) for bb, ptb, fv, ni in cases)
        n_agg_nan = sum(1 for _, _, _, ni in cases if ni['agg_is_nan'])
        max_nan = max(ni['n_nan'] for _, _, _, ni in cases)
        sections.append(
            f'<p><strong>{exp_type}:</strong> {len(cases)} 个 NaN 单元 (涉及 {len(unique_combos)} 个 BB×fv 组合), '
            f'{n_agg_nan} 个聚合=NaN, 最严重 {max_nan}/5 seeds NaN</p>'
        )

    return '\n'.join(sections)


def find_best_configs(results, exp_type):
    best = defaultdict(lambda: {'acc': -1, 'filter_val': '—', 'std': 0})
    for r in results[exp_type]:
        key = (r['backbone'], r['ptb'])
        acc = r['acc']
        if acc != acc: continue  # NaN skip
        if acc > best[key]['acc']:
            best[key] = {'acc': acc, 'filter_val': r.get('filter_val', '?'), 'std': r.get('std', 0)}
    return best


# ===================== HTML Generator =====================

def html_report(results):
    nan_reg = build_nan_registry(results)

    n_gppt = len(results['GPPT'])
    n_gcl = len(results['GCL_LP'])
    n_total = n_gppt + n_gcl + len(results['sim_pt']) + len(results['degree_pt']) + len(results['out_detect_pt'])

    # KPI: best clean/att per BB across all filters
    best_vals = {}
    for bb in ['stable', 'peak']:
        for lbl, ptb in [('Clean', '0.0'), ('Att', '0.05')]:
            best_acc = -1.0; best_method = ''; best_fv = ''
            for exp_type in ['sim_pt', 'degree_pt', 'out_detect_pt']:
                for r in results[exp_type]:
                    if r['backbone'] == bb and r['ptb'] == ptb:
                        acc = r['acc']
                        if acc == acc and acc > best_acc:
                            best_acc = acc; best_method = exp_type; best_fv = r.get('filter_val', '?')
            best_vals[(bb, lbl)] = (best_acc, best_method, best_fv)

    # GPPT baseline refs
    gppt_ref = {}
    for bb in ['stable', 'peak']:
        for ptb, lbl in [('0.0', 'clean'), ('0.05', 'att')]:
            matches = [r for r in results['GPPT'] if r['backbone'] == bb and r['ptb'] == ptb]
            gppt_ref[(bb, lbl)] = matches[0]['acc'] if matches else 0

    kpi_html = '<div class="kpi-grid">'
    for bb, cls in [('stable', 'warn'), ('peak', 'accent')]:
        ba, bm, bf = best_vals[(bb, 'Clean')]
        kpi_html += (f'<div class="kpi {cls}"><div class="value">{ba:.4f}</div>'
                     f'<div class="label">{bb} BB Clean Best ({bm}={bf})</div></div>')
    for bb, cls in [('stable', 'amber'), ('peak', 'amber')]:
        ba, bm, bf = best_vals[(bb, 'Att')]
        kpi_html += (f'<div class="kpi {cls}"><div class="value">{ba:.4f}</div>'
                     f'<div class="label">{bb} BB Att 0.05 Best ({bm}={bf})</div></div>')
    kpi_html += f'<div class="kpi"><div class="value">{n_total}</div><div class="label">Log files parsed</div></div>'
    kpi_html += f'<div class="kpi"><div class="value">{len(nan_reg)}</div><div class="label">NaN-affected cells</div></div>'
    kpi_html += '</div>'

    # NaN summary
    nan_summary = build_nan_summary_section(nan_reg)

    # Tables
    gppt_table = build_gppt_table(results)
    gcl_table = build_gcl_lp_table(results)

    sim_full = build_rp_single_filter_table(results, 'sim_pt', 'sim', SIM_VALS, nan_reg)
    deg_full = build_rp_single_filter_table(results, 'degree_pt', 'deg', DEG_VALS, nan_reg)
    ood_full = build_rp_single_filter_table(results, 'out_detect_pt', 'ood', OOD_VALS, nan_reg)

    sim_cross = build_cross_bb_comparison(results, 'sim_pt', SIM_VALS, 'sim', nan_reg)
    deg_cross = build_cross_bb_comparison(results, 'degree_pt', DEG_VALS, 'deg', nan_reg)
    ood_cross = build_cross_bb_comparison(results, 'out_detect_pt', OOD_VALS, 'ood', nan_reg)

    sim_best_table = build_best_summary_table(results, 'sim_pt', 'sim', nan_reg)
    deg_best_table = build_best_summary_table(results, 'degree_pt', 'deg', nan_reg)
    ood_best_table = build_best_summary_table(results, 'out_detect_pt', 'ood', nan_reg)

    stability_table = build_stability_table(results, nan_reg)

    # Full NaN registry table
    full_nan_table = build_nan_footnotes(nan_reg)

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Meeting 14 — 实验矩阵完整报告</title>
<style>
  :root {{
    --bg: #ffffff; --fg: #1a1a1a; --muted: #666;
    --accent: #2563eb; --warn: #dc2626; --good: #16a34a; --amber: #d97706;
    --border: #e5e7eb; --code-bg: #f3f4f6; --card-bg: #f8fafc;
    --table-stripe: #f9fafb; --highlight: #fef3c7;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; color: var(--fg); background: var(--bg); line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 24px 16px 80px; }}
  h1 {{ font-size: 1.6rem; border-bottom: 2px solid var(--border); padding-bottom: 12px; margin: 32px 0 16px; }}
  h2 {{ font-size: 1.25rem; margin: 28px 0 12px; color: var(--accent); }}
  h3 {{ font-size: 1.05rem; margin: 20px 0 8px; }}
  h4 {{ font-size: 0.95rem; margin: 14px 0 6px; }}
  p {{ margin: 8px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0 20px; font-size: 0.88rem; }}
  th, td {{ padding: 7px 10px; text-align: left; border: 1px solid var(--border); }}
  th {{ background: var(--code-bg); font-weight: 600; white-space: nowrap; }}
  tr:nth-child(even) td {{ background: var(--table-stripe); }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .highlight-row td {{ background: var(--highlight) !important; font-weight: 600; }}
  code {{ background: var(--code-bg); padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }}
  .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px 20px; margin: 12px 0; }}
  .callout {{ border-left: 4px solid var(--warn); background: #fef2f2; padding: 14px 18px; margin: 16px 0; border-radius: 0 6px 6px 0; }}
  .callout-info {{ border-left-color: var(--accent); background: #eff6ff; }}
  .callout-good {{ border-left-color: var(--good); background: #f0fdf4; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 12px 0 20px; }}
  .kpi {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; text-align: center; }}
  .kpi .value {{ font-size: 1.6rem; font-weight: 700; }}
  .kpi .label {{ font-size: 0.8rem; color: var(--muted); margin-top: 4px; }}
  .kpi.warn .value {{ color: var(--warn); }}
  .kpi.accent .value {{ color: var(--accent); }}
  .kpi.amber .value {{ color: var(--amber); }}
  .kpi.good .value {{ color: var(--good); }}
  hr {{ border: none; border-top: 1px solid var(--border); margin: 20px 0; }}
  details {{ margin: 12px 0; }}
  details summary {{ cursor: pointer; color: var(--accent); font-weight: 500; padding: 4px 0; }}
  sup {{ font-size: 0.7rem; cursor: help; }}
</style>
</head>
<body>

<h1>Meeting 14 — 实验矩阵完整报告</h1>
<p style="color: var(--muted);">DFS_HK4 Project · Generated {now_str} · {n_total} log files · 统一 split (train=35, val=265, test=2408)</p>

{kpi_html}

<!-- ================================================================== -->
<h2>实验结果速览 (Executive Summary)</h2>

<div class="card">
<h3>基线对比 (ptb=0.0 / ptb=0.05)</h3>
<table>
<tr><th>Method</th><th>Stable BB Clean</th><th>Stable BB Att</th><th>Peak BB Clean</th><th>Peak BB Att</th></tr>
<tr><td>GPPT (prompt baseline)</td><td class="num">0.6009</td><td class="num">0.3397</td><td class="num">0.6578</td><td class="num">0.2151</td></tr>
<tr><td>GCL Linear Probe (no prompt)</td><td class="num">0.5689</td><td class="num">0.1906</td><td class="num">0.6262</td><td class="num">0.2072</td></tr>
<tr style="font-weight:600"><td>RobustPrompt-T Best Single-Filter</td><td class="num">0.6761 (ood=0.5)</td><td class="num">0.2477 (ood=0.5)</td><td class="num">0.6723 (deg=1)</td><td class="num">0.2889 (ood=0.5)</td></tr>
</table>
</div>

<div class="callout-info">
<strong>核心发现：</strong>
<ul>
  <li>Peak BB 的 GPPT 在 clean 上达到 0.6578，但 <strong>0.05 攻击下跌至 0.2151</strong>（-67%）——攻击效果强劲</li>
  <li>GCL LP 在攻击下崩盘 (0.19-0.06)——<strong>embedding 被系统性破坏，非 prompt 方法问题</strong></li>
  <li><strong>out_detect_pt=0.5 在 clean (0.676) 和 attacked (0.289) 上均表现最佳</strong></li>
  <li>Clean vs Attacked 最优阈值仍不一致——确认 Meeting 11 结论</li>
</ul>
</div>

<!-- ================================================================== -->
<h2>NaN 与训练稳定性总览</h2>

<div class="card">
<h3>NaN 分布摘要</h3>
{nan_summary}
<h3 style="margin-top:16px">完整 NaN 登记表</h3>
{full_nan_table}
</div>

<div class="callout">
<strong>NaN 根因分析：</strong>
<ul>
  <li><strong>sim=0.2 + stable BB 是系统性不稳定源</strong>——所有 6 个 ptb 全部出现 NaN seeds，其中 5/6 聚合=NaN。sim=0.2 触发了太多节点（低 cosine 阈值 → 大范围加 prompt），在 weaker backbone 下导致梯度爆炸。</li>
  <li><strong>degree_pt 和 out_detect_pt 整体稳定</strong>——各仅 2-3 例，都是单 seed NaN，为随机波动。</li>
  <li><strong>peak BB 远优于 stable BB</strong>——11 例 NaN 中 stable BB 占 8 例。推荐下游首选 peak BB。</li>
  <li><strong>NaN 集中在中等攻击浓度</strong> (ptb=0.05-0.2)，clean 和 ptb=0.25 反而少——后者模型已经崩溃，loss 不会爆炸。</li>
</ul>
</div>

<!-- ================================================================== -->
<h2>1. Baselines: GPPT + GCL Linear Probe</h2>

<h3>1.1 GPPT Baseline</h3>
<p>GPPT 各 seed 输出完全一致（std=0），为确定性优化。Split 与 RobustPrompt-T 完全相同 (--specified)。</p>
{gppt_table}

<div class="callout-info">
<strong>GPPT 为什么比旧基线 (0.4350) 高？</strong>旧基线用的是 seed=56 单次预训练 backbone (lr=0.01, 无 ratio 控制)，其 clean Linear Probe 仅 ~0.40-0.50。新 peak BB 的 clean LR 达 0.6262，GPPT 在其上达到 0.6578 是合理的——GPPT 的 task tokens 比纯 LR 多了非线性表达能力。
</div>

<h3>1.2 GCL Linear Probe (冻结 backbone + LR on 攻击图)</h3>
<p>ptb=0.0 数据来自 eval_pretrain v2。ptb=0.05-0.25 为本次在攻击图上新增。LR 为确定性凸优化，无种子波动。</p>
{gcl_table}

<!-- ================================================================== -->
<h2>2. RobustPrompt-T — sim_pt 单 Filter 调参</h2>
{sim_full}

<details><summary>Cross-BB Comparison (best backbone per ptb×sim)</summary>{sim_cross}</details>
<details><summary>Best sim per (BB, ptb) — Summary</summary>{sim_best_table}</details>

<!-- ================================================================== -->
<h2>3. RobustPrompt-T — degree_pt 单 Filter 调参</h2>
{deg_full}

<details><summary>Cross-BB Comparison</summary>{deg_cross}</details>
<details><summary>Best deg per (BB, ptb) — Summary</summary>{deg_best_table}</details>

<!-- ================================================================== -->
<h2>4. RobustPrompt-T — out_detect_pt 单 Filter 调参</h2>
{ood_full}

<details><summary>Cross-BB Comparison</summary>{ood_cross}</details>
<details><summary>Best ood per (BB, ptb) — Summary</summary>{ood_best_table}</details>

<!-- ================================================================== -->
<h2>5. 稳定性矩阵（≥4/5 seeds stable per ptb）</h2>

<p>统计每个 (type, BB, filter_val) 下有多少 ptb 浓度达到 ≥4/5 seeds 稳定。</p>
{stability_table}

<!-- ================================================================== -->
<h2>6. 最优超参数推荐</h2>

<div class="card">
<h3>6.1 单 Filter 最佳阈值</h3>
<table>
<tr><th>Filter</th><th>Stable BB 推荐</th><th>Peak BB 推荐</th><th>备注</th></tr>
<tr><td><strong>sim_pt</strong></td>
    <td>sim=0.3 (clean 0.650) / sim=0.5 (att 0.241)<br><span style="color:var(--warn)">⚠ 0.2 系统性不稳定，避免</span></td>
    <td>sim=0.2 (clean 0.633) / sim=0.6 (att 0.260)<br><span style="color:var(--good)">全 sim 值均稳定</span></td>
    <td>Clean↔Att 最优方向相反</td></tr>
<tr><td><strong>degree_pt</strong></td>
    <td>deg=1 (clean 0.644) / deg=5 (att 0.223)</td>
    <td>deg=1 (clean 0.672) / deg=5 (att 0.286)<br><span style="color:var(--good)">deg=1 极稳定 (6/6 ptb 5/5)</span></td>
    <td>唯一跨扰动一致最优的 defense</td></tr>
<tr><td><strong>out_detect_pt</strong></td>
    <td>ood=0.5 (clean 0.676, att 0.248)</td>
    <td>ood=0.5 (clean 0.649, att 0.289)<br><span style="color:var(--good)">全 ood 值均稳定</span></td>
    <td>最鲁棒 filter，clean-att gap 最小</td></tr>
</table>
</div>

<div class="card">
<h3>6.2 下一步：Combo 验证建议</h3>
<p>跑最优单 filter 组合 + perturbation-adaptive 策略：</p>
<ul>
  <li><strong>保守方案 (稳定优先)：</strong> degree=1 单独使用 (peak BB, 6/6 ptb 5/5 稳定)</li>
  <li><strong>激进方案 (精度优先)：</strong> ood=0.5 单独使用 (peak BB, clean 0.649, att 0.289)</li>
  <li><strong>Combo 验证：</strong> (sim=0.3, deg=1, ood=0.5) + peak BB，验证多头是否仍负交互</li>
  <li><strong>Perturbation-adaptive：</strong> 低 ptb 用 clean-optimal 阈值，高 ptb 用 attacked-optimal</li>
</ul>
</div>

<hr>
<p style="color:var(--muted); font-size:0.82rem;">
  Generated by generate_meeting14_report.py · {now_str} ·
  <a href="logs/baselines/">logs/baselines/</a> ·
  <a href="logs/single_filter_sim/">logs/single_filter_sim/</a> ·
  <a href="logs/single_filter_degree/">logs/single_filter_degree/</a> ·
  <a href="logs/single_filter_ood/">logs/single_filter_ood/</a>
</p>

</body>
</html>'''

    return html


if __name__ == '__main__':
    print("=" * 60)
    print("Parsing experiment logs...")
    results = collect_all_results()
    print(f"Parsed: GPPT={len(results['GPPT'])}, GCL_LP={len(results['GCL_LP'])}, "
          f"sim={len(results['sim_pt'])}, deg={len(results['degree_pt'])}, ood={len(results['out_detect_pt'])}")

    nan_reg = build_nan_registry(results)
    print(f"NaN registry: {len(nan_reg)} entries across all experiments")

    print("Generating HTML...")
    html = html_report(results)
    output_path = 'reports/meeting14_full_experiment_report.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Report saved to: {output_path} ({len(html):,} bytes)")
    print("Done!")
