#!/usr/bin/env python3
"""5-filter combo HTML report — 4 data sources:
  (1) NSP p_plus=True   (2) NSP p_plus=False
  (3) NSP-IA p_plus=True  (4) NSP-IA p_plus=False (capped at 3-combo)
Generated: 2026-07-13."""

import os, re, json, numpy as np
from collections import defaultdict

def parse_dir(log_dir, max_combo_group=None):
    """Parse a log directory, return {combo: {ptb: {mean, std, n, seeds_str}}}.
    If max_combo_group is set, skip combos with more filters."""
    results = defaultdict(dict)
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith('.log'):
            continue
        m = re.match(r'(.+)_ptb([\d.]+)\.log$', fname)
        if not m:
            continue
        combo, ptb = m.group(1), float(m.group(2))

        # Filter by combo group size
        if max_combo_group is not None:
            n = 5 if combo == 'all5' else len(combo.split('+'))
            if n > max_combo_group:
                continue

        with open(os.path.join(log_dir, fname), 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        seeds = {}
        for sm in re.finditer(r'# Seed (\d+) Muti Split Final Acc: ([\d.]+|nan)', content):
            acc_str = sm.group(2)
            if acc_str != 'nan':
                seeds[int(sm.group(1))] = float(acc_str)
        v = list(seeds.values())
        entry = {
            'mean': float(np.mean(v)) if v else None,
            'std': float(np.std(v)) if len(v) > 1 else 0.0,
            'n': len(v),
            'seeds_str': ', '.join('{:.4f}'.format(x) for x in sorted(v))
        }
        # Guard against duplicate files (e.g. ptb0.1.log vs ptb0.10.log)
        # Keep the entry with more valid seeds
        prev = results[combo].get(ptb)
        if prev is None or len(v) > prev.get('n', 0):
            results[combo][ptb] = entry
    return dict(results)

PTBS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

def combo_group(c):
    if c == 'all5':
        return 5
    return len(c.split('+'))

def sort_key(c):
    return (combo_group(c), c)

SECTION_CSS = {1: 'section-single', 2: 'section-two', 3: 'section-three', 4: 'section-four', 5: 'section-five'}
SECTION_LABEL = {1: 'Single Filter', 2: 'Two-Filter', 3: 'Three-Filter', 4: 'Four-Filter', 5: 'Five-Filter'}

def safe_mean(rd, key, default=0.0):
    """Get mean from result dict, guarding against None values."""
    v = rd.get(key, {}).get('mean')
    return v if v is not None else default

def best_per_col(results):
    best = {}
    for p in PTBS:
        vals = [(c, r['mean']) for c, rd in results.items() if (r := rd.get(p)) and r['mean'] is not None]
        if vals:
            best[p] = max(v for _, v in vals)
    return best

def build_table_html(results, bp):
    rows_html = ''
    seed_notes = []
    for combo in sorted(results.keys(), key=sort_key):
        n = combo_group(combo)
        css = SECTION_CSS.get(n, '')
        rows_html += '<tr class="{}"><td>{}</td>'.format(css, combo)
        seed_parts = []
        for p in PTBS:
            r = results[combo].get(p, {})
            m = r.get('mean')
            s = r.get('std', 0)
            nv = r.get('n', 0)
            sd = r.get('seeds_str', '')
            is_best = m is not None and bp.get(p) is not None and abs(m - bp[p]) < 0.0005
            if m is None or nv == 0:
                rows_html += '<td class="na">--</td>'
                seed_parts.append('ptb={:.2f}: [no valid seeds]'.format(p))
            else:
                cell = '{:.4f}'.format(m)
                if s > 0.01:
                    cell += '&plusmn;{:.4f}'.format(s)
                if nv < 5:
                    cell += '*'
                if is_best:
                    cell = '<span class="best">{}</span>'.format(cell)
                rows_html += '<td>{}</td>'.format(cell)
                seed_parts.append('ptb={:.2f}: [{}]'.format(p, sd))
        rows_html += '</tr>\n'
        seed_notes.append((combo, '; '.join(seed_parts)))
    return rows_html, seed_notes

def build_seed_html(seed_notes):
    html = ''
    for combo, detail in seed_notes:
        html += '<div class="seed-detail"><strong>{}:</strong> {}</div>\n'.format(combo, detail)
    return html

def build_best_rows(results, bp):
    html = ''
    for p in PTBS:
        ranked = [(c, r['mean']) for c, rd in results.items() if (r := rd.get(p)) and r['mean'] is not None]
        ranked.sort(key=lambda x: -x[1])
        if ranked:
            r1, r2 = ranked[0], ranked[1] if len(ranked) > 1 else (None, None)
            html += '<tr><td>{:.2f}</td><td><strong>{}</strong></td><td><strong>{:.4f}</strong></td>'.format(
                p, r1[0], r1[1])
            if r2:
                html += '<td>{} ({:.4f})</td>'.format(r2[0], r2[1])
            html += '</tr>\n'
    return html

def build_single_best_table(results):
    """For single filters only: rank them per ptb."""
    singles = {c: rd for c, rd in results.items() if combo_group(c) == 1}
    html = '<table class="data-table"><thead><tr><th>ptb</th>'
    ranked_per_ptb = {}
    for p in PTBS:
        ranked = [(c, r['mean']) for c, rd in singles.items() if (r := rd.get(p)) and r['mean'] is not None]
        ranked.sort(key=lambda x: -x[1])
        ranked_per_ptb[p] = ranked
    filter_names = sorted(singles.keys())
    for fn in filter_names:
        html += '<th>{}</th>'.format(fn)
    html += '</tr></thead><tbody>\n'
    for p in PTBS:
        html += '<tr><td>{:.2f}</td>'.format(p)
        best_val = max((v for _, v in ranked_per_ptb[p]), default=None)
        vals = {c: v for c, v in ranked_per_ptb[p]}
        for fn in filter_names:
            v = vals.get(fn)
            if v is None:
                html += '<td class="na">--</td>'
            elif best_val is not None and abs(v - best_val) < 0.0005:
                html += '<td><span class="best">{:.4f}</span></td>'.format(v)
            else:
                html += '<td>{:.4f}</td>'.format(v)
        html += '</tr>\n'
    html += '</tbody></table>\n'
    return html

def build_robustness_summary(results, label):
    """Compute clean acc and worst-case ptb acc for each combo group size."""
    max_n = max((combo_group(c) for c in results.keys()), default=3)
    html = '<h3>{} — Robustness Summary (clean → M-0.25)</h3>\n'.format(label)
    html += '<table class="data-table"><thead><tr><th>Group Size</th><th>Best Clean</th><th>Best M-0.25</th><th>Drop</th></tr></thead><tbody>\n'
    for n in range(1, max_n + 1):
        group = {c: rd for c, rd in results.items() if combo_group(c) == n}
        clean_best = max((safe_mean(r, 0.0) for r in group.values()), default=0)
        m25_best = max((safe_mean(r, 0.25) for r in group.values()), default=0)
        drop = clean_best - m25_best
        html += '<tr><td>{}-filter</td><td>{:.4f}</td><td>{:.4f}</td><td>{:.4f}</td></tr>\n'.format(n, clean_best, m25_best, drop)
    html += '</tbody></table>\n'
    return html

def build_pplus_comparison_rows(results_pp, results_nopp, bp_pp, bp_nopp, shared_combos):
    """Side-by-side p_plus=True vs p_plus=False for matching combos."""
    html = ''
    for combo in sorted(shared_combos, key=sort_key):
        n = combo_group(combo)
        css = SECTION_CSS.get(n, '')
        html += '<tr class="{}"><td>{}</td>'.format(css, combo)
        for p in PTBS:
            r_pp = results_pp.get(combo, {}).get(p, {})
            r_np = results_nopp.get(combo, {}).get(p, {})
            m_pp = r_pp.get('mean')
            m_np = r_np.get('mean')

            if m_pp is None and m_np is None:
                html += '<td class="na">--</td>'
            elif m_np is None:
                html += '<td>{:.4f} vs --</td>'.format(m_pp)
            elif m_pp is None:
                html += '<td>-- vs {:.4f}</td>'.format(m_np)
            else:
                diff = m_np - m_pp
                cls = 'diff-pos' if diff > 0.005 else ('diff-neg' if diff < -0.005 else '')
                cell = '{:.4f} vs {:.4f}'.format(m_pp, m_np)
                if cls:
                    dsign = '+' if diff > 0 else ''
                    cell += ' <span class="{}">({}{:.4f})</span>'.format(cls, dsign, diff)
                html += '<td>{}</td>'.format(cell)
        html += '</tr>\n'
    return html


# ============================================================
# MAIN
# ============================================================
print("Parsing logs/5filter_combos (NSP true-label, p_plus=True)...")
results_nsp = parse_dir('logs/5filter_combos')
print("  {} combos, {} total entries".format(len(results_nsp), sum(len(v) for v in results_nsp.values())))

print("Parsing logs/5filter_combos_ia (NSP-IA pseudo-label, p_plus=True)...")
results_ia = parse_dir('logs/5filter_combos_ia')
print("  {} combos, {} total entries".format(len(results_ia), sum(len(v) for v in results_ia.values())))

print("Parsing logs/5filter_combos_nopplus (NSP true-label, p_plus=False)...")
results_nsp_np = parse_dir('logs/5filter_combos_nopplus')
print("  {} combos, {} total entries".format(len(results_nsp_np), sum(len(v) for v in results_nsp_np.values())))

print("Parsing logs/5filter_combos_ia_nopplus (NSP-IA pseudo-label, p_plus=False, capped at 3-combo)...")
results_ia_np = parse_dir('logs/5filter_combos_ia_nopplus', max_combo_group=3)
print("  {} combos, {} total entries".format(len(results_ia_np), sum(len(v) for v in results_ia_np.values())))

bp_nsp = best_per_col(results_nsp)
bp_ia = best_per_col(results_ia)
bp_nsp_np = best_per_col(results_nsp_np)
bp_ia_np = best_per_col(results_ia_np)

# Build table rows
rows_nsp, seed_nsp = build_table_html(results_nsp, bp_nsp)
rows_ia, seed_ia = build_table_html(results_ia, bp_ia)
rows_nsp_np, seed_nsp_np = build_table_html(results_nsp_np, bp_nsp_np)
rows_ia_np, seed_ia_np = build_table_html(results_ia_np, bp_ia_np)

# p_plus comparison: shared combos only
shared_nsp = set(results_nsp.keys()) & set(results_nsp_np.keys())
shared_ia = set(results_ia.keys()) & set(results_ia_np.keys())
comp_pp_nsp_rows = build_pplus_comparison_rows(results_nsp, results_nsp_np, bp_nsp, bp_nsp_np, shared_nsp)
comp_pp_ia_rows = build_pplus_comparison_rows(results_ia, results_ia_np, bp_ia, bp_ia_np, shared_ia)

# ============================================================
# HTML
# ============================================================
html = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>5-Filter Combo — Full Report with p_plus Ablation</title>
<style>
  :root {
    --bg: #f8fafc;
    --surface: #ffffff;
    --border: #e2e8f0;
    --text: #1e293b;
    --text-secondary: #64748b;
    --text-muted: #94a3b8;
    --accent: #2563eb;
    --accent-light: #eff6ff;
    --accent-hover: #1d4ed8;
    --green: #059669;
    --green-bg: #ecfdf5;
    --green-border: #a7f3d0;
    --red: #dc2626;
    --red-bg: #fef2f2;
    --red-border: #fecaca;
    --amber: #d97706;
    --amber-bg: #fffbeb;
    --amber-border: #fde68a;
    --purple: #7c3aed;
    --purple-bg: #f5f3ff;
    --purple-border: #ddd6fe;
    --shadow-xs: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 6px rgba(0,0,0,0.05), 0 2px 4px rgba(0,0,0,0.04);
    --radius: 8px;
    --radius-sm: 5px;
    --radius-xs: 3px;
  }

  * { margin:0; padding:0; box-sizing:border-box; }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 13px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  .container {
    max-width: 1540px;
    margin: 0 auto;
    padding: 36px 28px 48px;
  }

  /* ── Header ── */
  header {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 26px 30px;
    margin-bottom: 28px;
    box-shadow: var(--shadow-sm);
  }
  header h1 {
    font-size: 21px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.012em;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
  }
  header .meta {
    font-size: 11.5px;
    color: var(--text-secondary);
    line-height: 1.75;
  }
  header .meta code {
    background: #f1f5f9;
    padding: 1px 6px;
    border-radius: var(--radius-xs);
    font-size: 10.5px;
    color: var(--text);
  }

  /* ── Section headings ── */
  h2 {
    font-size: 15px;
    font-weight: 650;
    color: var(--text);
    margin: 36px 0 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--accent);
    letter-spacing: -0.008em;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  h3 {
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text);
    margin: 26px 0 10px;
    letter-spacing: -0.005em;
  }

  /* ── Tables ── */
  table.data-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 11.5px;
    font-variant-numeric: tabular-nums;
    margin: 8px 0 20px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow-xs);
  }
  table.data-table thead th {
    background: #f1f5f9;
    font-weight: 600;
    text-align: center;
    padding: 9px 10px;
    border-bottom: 2px solid #cbd5e1;
    white-space: nowrap;
    color: var(--text);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.035em;
    position: sticky;
    top: 0;
    z-index: 2;
  }
  table.data-table thead th:first-child {
    text-align: left;
    padding-left: 16px;
  }
  table.data-table tbody td {
    text-align: center;
    padding: 7px 10px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
    transition: background 0.1s ease;
  }
  table.data-table tbody td:first-child {
    text-align: left;
    padding-left: 16px;
    font-weight: 500;
    font-size: 11.5px;
  }
  table.data-table tbody tr:last-child td {
    border-bottom: none;
  }
  table.data-table tbody tr:hover td {
    background: #f8fafc !important;
  }

  /* ── Filter-level row shading ── */
  .section-single { background: #ffffff; }
  .section-two    { background: #f5f7ff; }
  .section-three  { background: #eef2ff; }
  .section-four   { background: #e4ebfc; }
  .section-five   { background: #d9e2fa; font-weight: 700; }

  /* ── Best / NA ── */
  .best {
    font-weight: 700;
    color: var(--green);
    background: var(--green-bg);
    padding: 2px 7px;
    border-radius: var(--radius-xs);
    display: inline-block;
  }
  .na {
    color: #cbd5e1;
    font-style: italic;
    font-weight: 400;
  }

  /* ── Notes ── */
  .note {
    font-size: 11px;
    color: var(--text-secondary);
    margin: 4px 0 10px;
    line-height: 1.65;
  }

  /* ── Seed detail blocks ── */
  .seed-detail {
    font-size: 10.5px;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
    margin: 3px 0;
    padding: 5px 10px;
    background: #f8fafc;
    border-radius: var(--radius-xs);
    line-height: 1.7;
    word-break: break-all;
  }

  /* ── Expandable details ── */
  details {
    margin: 10px 0 18px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }
  details summary {
    font-size: 12px;
    font-weight: 550;
    color: var(--accent);
    cursor: pointer;
    user-select: none;
    padding: 10px 16px;
    background: #f8fafc;
    transition: background 0.12s ease;
  }
  details summary:hover {
    background: #f1f5f9;
    color: var(--accent-hover);
  }
  details[open] summary {
    border-bottom: 1px solid var(--border);
  }
  details .seed-detail {
    margin: 0;
    border-radius: 0;
    border-bottom: 1px solid #f1f5f9;
  }
  details .seed-detail:last-child {
    border-bottom: none;
  }

  /* ── Callout boxes ── */
  .finding, .bad, .good, .warn {
    padding: 13px 18px;
    margin: 10px 0;
    font-size: 12px;
    border-radius: var(--radius);
    border-left: 4px solid;
    line-height: 1.65;
    box-shadow: var(--shadow-xs);
  }
  .finding {
    background: #f0f4ff;
    border-left-color: var(--accent);
  }
  .bad {
    background: var(--red-bg);
    border-left-color: var(--red);
  }
  .good {
    background: var(--green-bg);
    border-left-color: var(--green);
  }
  .warn {
    background: var(--amber-bg);
    border-left-color: var(--amber);
  }
  .finding code, .bad code, .good code, .warn code {
    background: rgba(0,0,0,0.06);
    padding: 1px 5px;
    border-radius: var(--radius-xs);
    font-size: 10.5px;
  }
  .finding strong, .bad strong, .good strong, .warn strong {
    color: var(--text);
  }

  hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 26px 0;
  }

  /* ── Diff markers ── */
  .diff-pos {
    font-weight: 700;
    color: var(--green);
    background: var(--green-bg);
    padding: 1px 5px;
    border-radius: var(--radius-xs);
  }
  .diff-neg {
    font-weight: 700;
    color: var(--red);
    background: var(--red-bg);
    padding: 1px 5px;
    border-radius: var(--radius-xs);
  }

  /* ── Tab navigation ── */
  .tab-nav {
    display: flex;
    gap: 3px;
    margin-bottom: 26px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 5px;
    box-shadow: var(--shadow-sm);
    flex-wrap: wrap;
  }
  .tab-btn {
    padding: 8px 18px;
    cursor: pointer;
    border: none;
    background: transparent;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
    border-radius: 6px;
    white-space: nowrap;
    transition: all 0.15s ease;
    font-family: inherit;
  }
  .tab-btn:hover {
    background: #f1f5f9;
    color: var(--text);
  }
  .tab-btn.active {
    background: var(--accent);
    color: #ffffff;
    font-weight: 600;
    box-shadow: var(--shadow);
  }
  .tab-panel {
    display: none;
  }
  .tab-panel.active {
    display: block;
    animation: tabFadeIn 0.18s ease;
  }
  @keyframes tabFadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* ── Badges ── */
  .fix-badge, .new-badge, .cap-badge {
    display: inline-flex;
    align-items: center;
    font-size: 10px;
    padding: 2px 9px;
    border-radius: 12px;
    margin-left: 8px;
    font-weight: 600;
    letter-spacing: 0.02em;
    vertical-align: middle;
    line-height: 1.5;
  }
  .fix-badge {
    background: var(--green-bg);
    color: #047857;
    border: 1px solid var(--green-border);
  }
  .new-badge {
    background: var(--purple-bg);
    color: #6d28d9;
    border: 1px solid var(--purple-border);
  }
  .cap-badge {
    background: var(--amber-bg);
    color: #b45309;
    border: 1px solid var(--amber-border);
  }

  /* ── Metric boxes ── */
  .metric-box {
    display: inline-block;
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 8px 16px;
    margin: 4px;
    border-radius: var(--radius);
    font-size: 12px;
    box-shadow: var(--shadow-xs);
  }
  .metric-box strong {
    color: var(--accent);
    font-weight: 650;
  }

  /* ── Legend swatches ── */
  .legend-swatch {
    display: inline-block;
    padding: 2px 10px;
    border-radius: var(--radius-xs);
    font-size: 10.5px;
    font-weight: 500;
    margin: 0 1px;
  }
  .legend-2f { background: #f5f7ff; }
  .legend-3f { background: #eef2ff; }
  .legend-4f { background: #e4ebfc; }
  .legend-5f { background: #d9e2fa; font-weight: 650; }

  /* ── Responsive ── */
  @media (max-width: 900px) {
    .container { padding: 16px 12px; }
    header { padding: 16px 18px; }
    table.data-table { font-size: 10px; }
    table.data-table thead th,
    table.data-table tbody td { padding: 4px 5px; }
    .tab-btn { padding: 6px 10px; font-size: 10px; }
  }

  /* ── Print ── */
  @media print {
    body { background: #fff; font-size: 10px; }
    .tab-panel { display: block !important; page-break-inside: avoid; }
    .tab-nav { display: none; }
    header { box-shadow: none; border: 1px solid #ddd; }
    table.data-table { box-shadow: none; }
    .finding, .bad, .good, .warn { box-shadow: none; }
  }
</style>
</head>
<body>
<div class="container">

<header>
  <h1>5-Filter Combo Experiment Report <span class="fix-badge">CORRECTED</span> <span class="new-badge">p_plus ABLATION</span></h1>
  <div class="meta">
    Generated: 2026-07-13 &nbsp;|&nbsp;
    Peak BB: permE-maskN lr=0.001 r=0.3 seed=1 (256-dim GCN) &nbsp;|&nbsp;
    4 data sources: NSP/NSP-IA × p_plus=True/False &nbsp;|&nbsp;
    5 seeds per experiment, mean &plusmn; std (no min removal)
  </div>
  <div class="meta" style="margin-top:4px;">
    <strong>Fixes applied:</strong> (1) <code>task.py</code> filters negative-threshold entries from
    <code>muti_defense_pt_dict</code> before prompt construction — threshold=-1 truly disables.
    (2) p_plus=False uses single prompt token per tip (~256 params) vs p_plus=True uses
    20-token bank + learned combination (~10K params per filter).
  </div>
</header>

<!-- ============ TAB NAV ============ -->
<div class="tab-nav">
  <button class="tab-btn active" data-tab="nsp" onclick="showTab('nsp')">NSP (p_plus=True)</button>
  <button class="tab-btn" data-tab="nsp_np" onclick="showTab('nsp_np')">NSP (p_plus=False) <span class="new-badge" style="margin-left:4px;">NEW</span></button>
  <button class="tab-btn" data-tab="ia" onclick="showTab('ia')">NSP-IA (p_plus=True)</button>
  <button class="tab-btn" data-tab="ia_np" onclick="showTab('ia_np')">NSP-IA (p_plus=False) <span class="cap-badge" style="margin-left:4px;">1-3F</span></button>
  <button class="tab-btn" data-tab="compare_pp" onclick="showTab('compare_pp')">p_plus=True vs False</button>
  <button class="tab-btn" data-tab="analysis" onclick="showTab('analysis')">Analysis</button>
</div>

<script>
function showTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  const btn = document.querySelector('[data-tab="' + name + '"]');
  if (btn) btn.classList.add('active');
}
</script>
'''

# ================================================================
# Helper: render a full tab panel with accuracy table + best + robustness + singles
# ================================================================
def render_tab_panel(tab_id, title, subtitle, results, bp, rows_html, seed_html, extra_note=''):
    out = '<div id="tab-{}" class="tab-panel{}">\n'.format(tab_id, ' active' if tab_id == 'nsp' else '')
    out += '<h2>{}</h2>\n'.format(title)
    out += '<p class="note">{}</p>\n'.format(subtitle)
    if extra_note:
        out += extra_note + '\n'
    out += '<p class="note">\n'
    out += '  <span class="legend-swatch legend-2f">2-filter</span> '
    out += '<span class="legend-swatch legend-3f">3-filter</span> '
    out += '<span class="legend-swatch legend-4f">4-filter</span> '
    out += '<span class="legend-swatch legend-5f">5-filter</span>. '
    out += '<span class="best">Green bold</span> = best in column. '
    out += '<code>--</code> = all seeds NaN. <code>*</code> = fewer than 5 valid seeds.\n'
    out += '</p>\n'
    out += '<table class="data-table">\n'
    out += '<thead><tr><th>Combo</th>'
    for p in PTBS:
        out += '<th>{:.2f}</th>'.format(p)
    out += '</tr></thead><tbody>\n'
    out += rows_html
    out += '</tbody></table>\n'

    out += '<details>\n<summary>Per-combo seed details (click to expand)</summary>\n'
    out += seed_html
    out += '</details>\n'

    out += '<h3>Best Per Perturbation Level</h3>\n'
    out += '<table class="data-table"><thead><tr><th>ptb</th><th>Best Combo</th><th>Accuracy</th><th>Runner-up</th></tr></thead><tbody>\n'
    out += build_best_rows(results, bp)
    out += '</tbody></table>\n'

    out += build_robustness_summary(results, title)

    out += '<h3>Single Filter Comparison</h3>\n'
    out += build_single_best_table(results)

    out += '</div><!-- end tab-{} -->\n'.format(tab_id)
    return out


# ---- Tab 1: NSP p_plus=True ----
html += render_tab_panel('nsp',
    'NSP — True Labels (p_plus=True)',
    'p_plus=True: 20-token bank + learned combination (~10K params per filter). Baseline from previous corrected report.',
    results_nsp, bp_nsp, rows_nsp, build_seed_html(seed_nsp))

# ---- Tab 2: NSP p_plus=False ----
html += render_tab_panel('nsp_np',
    'NSP — True Labels (p_plus=False) <span class="new-badge">NEW</span>',
    'p_plus=False: single prompt token per tip (~256 params per filter). loss_constraint enabled. 186 total runs.',
    results_nsp_np, bp_nsp_np, rows_nsp_np, build_seed_html(seed_nsp_np))

# ---- Tab 3: NSP-IA p_plus=True ----
html += render_tab_panel('ia',
    'NSP-IA — Pseudo-Labels (p_plus=True)',
    'IA-PT pseudo-label expansion (35→~595 labels) + p_plus=True token bank.',
    results_ia, bp_ia, rows_ia, build_seed_html(seed_ia))

# ---- Tab 4: NSP-IA p_plus=False ----
html += '<div id="tab-ia_np" class="tab-panel">\n'
html += '<h2>NSP-IA — Pseudo-Labels (p_plus=False) <span class="cap-badge">1-3 FILTERS ONLY</span></h2>\n'
html += '<p class="note">IA-PT pseudo-label expansion + p_plus=False. '
html += '4-filter and 5-filter combos had incomplete data (0-3 valid seeds) and are excluded from this tab. '
html += 'Only 1/2/3-filter combos (25 groups × 6 ptb = 150 runs) are shown.</p>\n'
html += '<p class="note">'
html += '  <span class="legend-swatch legend-2f">2-filter</span> '
html += '<span class="legend-swatch legend-3f">3-filter</span>. '
html += '<span class="best">Green bold</span> = best in column. '
html += '<code>--</code> = all seeds NaN. <code>*</code> = fewer than 5 valid seeds.\n'
html += '</p>\n'
html += '<table class="data-table">\n'
html += '<thead><tr><th>Combo</th>'
for p in PTBS:
    html += '<th>{:.2f}</th>'.format(p)
html += '</tr></thead><tbody>\n'
html += rows_ia_np
html += '</tbody></table>\n'

html += '<details>\n<summary>Per-combo seed details (click to expand)</summary>\n'
html += build_seed_html(seed_ia_np)
html += '</details>\n'

html += '<h3>Best Per Perturbation Level</h3>\n'
html += '<table class="data-table"><thead><tr><th>ptb</th><th>Best Combo</th><th>Accuracy</th><th>Runner-up</th></tr></thead><tbody>\n'
html += build_best_rows(results_ia_np, bp_ia_np)
html += '</tbody></table>\n'

html += build_robustness_summary(results_ia_np, 'NSP-IA p_plus=False')

html += '<h3>Single Filter Comparison</h3>\n'
html += build_single_best_table(results_ia_np)

html += '</div><!-- end tab-ia_np -->\n'

# ---- Tab 5: p_plus=True vs False comparison ----
html += '''<div id="tab-compare_pp" class="tab-panel">

<h2>p_plus=True vs p_plus=False — Comparison</h2>
<p class="note">
  Each cell: <strong>p_plus=True</strong> vs <strong>p_plus=False</strong>.
  <span class="diff-pos">Green delta</span> = p_plus=False <em>better</em> by &gt;0.005.
  <span class="diff-neg">Red delta</span> = p_plus=False <em>worse</em> by &gt;0.005.
  Shared combos only; missing data shown as <code>--</code>.
</p>

<h3>NSP True-Label: p_plus=True vs p_plus=False</h3>
<table class="data-table">
<thead><tr><th>Combo</th>'''
for p in PTBS:
    html += '<th>{:.2f}</th>'.format(p)
html += '</tr></thead><tbody>\n'
html += comp_pp_nsp_rows
html += '</tbody></table>\n'

# NSP p_plus win/loss
html += '<h3>NSP — p_plus=False Win/Loss vs p_plus=True</h3>\n'
html += '<table class="data-table"><thead><tr><th>ptb</th><th>p_plus=False Wins</th><th>p_plus=False Losses</th><th>Ties</th><th>Avg Delta (np−pp)</th></tr></thead><tbody>\n'
for p in PTBS:
    wins = losses = ties = 0
    deltas = []
    for combo in sorted(shared_nsp):
        r_pp = results_nsp[combo].get(p, {})
        r_np = results_nsp_np[combo].get(p, {})
        m_pp, m_np = r_pp.get('mean'), r_np.get('mean')
        if m_pp is None or m_np is None:
            continue
        d = m_np - m_pp
        deltas.append(d)
        if d > 0.005:
            wins += 1
        elif d < -0.005:
            losses += 1
        else:
            ties += 1
    avg_d = np.mean(deltas) if deltas else 0
    html += '<tr><td>{:.2f}</td><td>{}</td><td>{}</td><td>{}</td><td>{:+.4f}</td></tr>\n'.format(p, wins, losses, ties, avg_d)
html += '</tbody></table>\n'

# NSP-IA p_plus win/loss
html += '<h3>NSP-IA — p_plus=False Win/Loss vs p_plus=True (1-3F shared only)</h3>\n'
html += '<table class="data-table"><thead><tr><th>ptb</th><th>p_plus=False Wins</th><th>p_plus=False Losses</th><th>Ties</th><th>Avg Delta (np−pp)</th></tr></thead><tbody>\n'
for p in PTBS:
    wins = losses = ties = 0
    deltas = []
    for combo in sorted(shared_ia):
        r_pp = results_ia[combo].get(p, {})
        r_np = results_ia_np[combo].get(p, {})
        m_pp, m_np = r_pp.get('mean'), r_np.get('mean')
        if m_pp is None or m_np is None:
            continue
        d = m_np - m_pp
        deltas.append(d)
        if d > 0.005:
            wins += 1
        elif d < -0.005:
            losses += 1
        else:
            ties += 1
    avg_d = np.mean(deltas) if deltas else 0
    html += '<tr><td>{:.2f}</td><td>{}</td><td>{}</td><td>{}</td><td>{:+.4f}</td></tr>\n'.format(p, wins, losses, ties, avg_d)
html += '</tbody></table>\n'

# IA side-by-side comparison
html += '<h3>NSP-IA: p_plus=True vs p_plus=False — Side-by-Side (1-3F shared)</h3>\n'
html += '<table class="data-table">\n'
html += '<thead><tr><th>Combo</th>'
for p in PTBS:
    html += '<th>{:.2f}</th>'.format(p)
html += '</tr></thead><tbody>\n'
html += comp_pp_ia_rows
html += '</tbody></table>\n'

html += '</div><!-- end tab-compare_pp -->\n'


# ================================================================
# TAB: ANALYSIS (updated with p_plus ablation)
# ================================================================
def find_best(results, ptb_val):
    best_c, best_v = None, -1
    for c, rd in results.items():
        r = rd.get(ptb_val, {})
        if r.get('mean') and r['mean'] > best_v:
            best_v = r['mean']
            best_c = c
    return best_c, best_v

def find_worst(results, ptb_val):
    worst_c, worst_v = None, 999
    for c, rd in results.items():
        r = rd.get(ptb_val, {})
        if r.get('mean') is not None and r['mean'] < worst_v:
            worst_v = r['mean']
            worst_c = c
    return worst_c, worst_v

def best_per_group(results, ptb):
    groups = defaultdict(list)
    for c, rd in results.items():
        r = rd.get(ptb, {})
        if r.get('mean'):
            groups[combo_group(c)].append((c, r['mean']))
    out = {}
    max_n = max(groups.keys()) if groups else 3
    for n in range(1, max_n + 1):
        if groups[n]:
            out[n] = max(groups[n], key=lambda x: x[1])
    return out

# ---- Key metrics ----
# NSP p_plus=True
nsp_best_clean = find_best(results_nsp, 0.0)
nsp_best_m25 = find_best(results_nsp, 0.25)
# NSP p_plus=False
np_best_clean = find_best(results_nsp_np, 0.0)
np_best_m25 = find_best(results_nsp_np, 0.25)
# IA p_plus=True
ia_best_clean = find_best(results_ia, 0.0)
ia_best_m25 = find_best(results_ia, 0.25)
# IA p_plus=False
ianp_best_clean = find_best(results_ia_np, 0.0)
ianp_best_m25 = find_best(results_ia_np, 0.25)

# Single filter rankings
nsp_singles = [(c, safe_mean(rd, 0.0)) for c, rd in results_nsp.items() if combo_group(c) == 1]
nsp_singles.sort(key=lambda x: -x[1])
np_singles = [(c, safe_mean(rd, 0.0)) for c, rd in results_nsp_np.items() if combo_group(c) == 1]
np_singles.sort(key=lambda x: -x[1])
ia_singles = [(c, safe_mean(rd, 0.0)) for c, rd in results_ia.items() if combo_group(c) == 1]
ia_singles.sort(key=lambda x: -x[1])
ianp_singles = [(c, safe_mean(rd, 0.0)) for c, rd in results_ia_np.items() if combo_group(c) == 1]
ianp_singles.sort(key=lambda x: -x[1])

# Clean/M-0.25 for NSP groups
best_single_clean_nsp = max((safe_mean(rd, 0.0) for c, rd in results_nsp.items() if combo_group(c)==1), default=0)
best_multi_clean_nsp = max((safe_mean(rd, 0.0) for c, rd in results_nsp.items() if combo_group(c)>1), default=0)
best_single_m25_nsp = max((safe_mean(rd, 0.25) for c, rd in results_nsp.items() if combo_group(c)==1), default=0)
best_multi_m25_nsp = max((safe_mean(rd, 0.25) for c, rd in results_nsp.items() if combo_group(c)>1), default=0)

# Same for p_plus=False
best_single_clean_np = max((safe_mean(rd, 0.0) for c, rd in results_nsp_np.items() if combo_group(c)==1), default=0)
best_multi_clean_np = max((safe_mean(rd, 0.0) for c, rd in results_nsp_np.items() if combo_group(c)>1), default=0)
best_single_m25_np = max((safe_mean(rd, 0.25) for c, rd in results_nsp_np.items() if combo_group(c)==1), default=0)
best_multi_m25_np = max((safe_mean(rd, 0.25) for c, rd in results_nsp_np.items() if combo_group(c)>1), default=0)

# NaN counts
sim_nan_count = sum(1 for p in PTBS if results_nsp.get('sim', {}).get(p, {}).get('n', 0) == 0)
sim_nan_count_np = sum(1 for p in PTBS if results_nsp_np.get('sim', {}).get(p, {}).get('n', 0) == 0)

# best m25 per combo group
nsp_m25_groups = best_per_group(results_nsp, 0.25)
np_m25_groups = best_per_group(results_nsp_np, 0.25)

# Drops
nsp_clean_to_m25 = nsp_best_clean[1] - nsp_best_m25[1] if nsp_best_clean[0] and nsp_best_m25[0] else 0
np_clean_to_m25 = np_best_clean[1] - np_best_m25[1] if np_best_clean[0] and np_best_m25[0] else 0
ia_clean_to_m25 = ia_best_clean[1] - ia_best_m25[1] if ia_best_clean[0] and ia_best_m25[0] else 0
ianp_clean_to_m25 = ianp_best_clean[1] - ianp_best_m25[1] if ianp_best_clean[0] and ianp_best_m25[0] else 0

# Clean mean delta for p_plus=True→False
def avg_pplus_delta(results_pp, results_np, ptb, combos):
    deltas = []
    for c in combos:
        m_pp = results_pp.get(c, {}).get(ptb, {}).get('mean')
        m_np = results_np.get(c, {}).get(ptb, {}).get('mean')
        if m_pp is not None and m_np is not None:
            deltas.append(m_np - m_pp)
    return np.mean(deltas) if deltas else 0

avg_clean_delta_nsp = avg_pplus_delta(results_nsp, results_nsp_np, 0.0, shared_nsp)
avg_m25_delta_nsp = avg_pplus_delta(results_nsp, results_nsp_np, 0.25, shared_nsp)
avg_clean_delta_ia = avg_pplus_delta(results_ia, results_ia_np, 0.0, shared_ia)
avg_m25_delta_ia = avg_pplus_delta(results_ia, results_ia_np, 0.25, shared_ia)

# Best single filter per dataset
nsp_best_single_name = nsp_singles[0][0] if nsp_singles else '?'
nsp_best_single_val = nsp_singles[0][1] if nsp_singles else 0
np_best_single_name = np_singles[0][0] if np_singles else '?'
np_best_single_val = np_singles[0][1] if np_singles else 0

# lowest 2-filter
nsp_2f = [(c, safe_mean(rd, 0.0)) for c, rd in results_nsp.items() if combo_group(c)==2]
nsp_2f.sort(key=lambda x: x[1])
worst_2f_name = nsp_2f[0][0] if nsp_2f else '?'
worst_2f_val = nsp_2f[0][1] if nsp_2f else 0

# fc at M-0.25
fc_m25_nsp = safe_mean(results_nsp.get('fc', {}), 0.25)
fc_m25_np = safe_mean(results_nsp_np.get('fc', {}), 0.25)
nsp_clean_val = safe_mean(results_nsp.get('nsp', {}), 0.0)
deg_clean_val = safe_mean(results_nsp.get('degree', {}), 0.0)
fc_clean_val = safe_mean(results_nsp.get('fc', {}), 0.0)
nsp_clean_np = safe_mean(results_nsp_np.get('nsp', {}), 0.0)
deg_clean_np = safe_mean(results_nsp_np.get('degree', {}), 0.0)
fc_clean_np = safe_mean(results_nsp_np.get('fc', {}), 0.0)

# Average rank change for p_plus False vs True
def avg_rank_change(results_pp, results_np, ptb, shared):
    pp_ranked = sorted([(c, rd.get(ptb,{}).get('mean', -1)) for c, rd in results_pp.items() if c in shared],
                        key=lambda x: -x[1])
    np_ranked = sorted([(c, rd.get(ptb,{}).get('mean', -1)) for c, rd in results_np.items() if c in shared],
                        key=lambda x: -x[1])
    pp_rank = {c: i for i, (c, _) in enumerate(pp_ranked)}
    np_rank = {c: i for i, (c, _) in enumerate(np_ranked)}
    changes = [abs(pp_rank.get(c, 0) - np_rank.get(c, 0)) for c in shared]
    return np.mean(changes) if changes else 0

avg_rank_shift = avg_rank_change(results_nsp, results_nsp_np, 0.0, shared_nsp)

analysis_html = '''
<h2>Analysis & Key Findings</h2>

<div class="good"><strong>1. Fix verified — focusedcleaner_pt now properly disabled.</strong>
With the <code>task.py</code> fix (threshold &lt; 0 → key removed from <code>pt_dict</code>),
all 31 combos × 6 ptb now correctly isolate individual filter contributions.
This corrected data serves as the <code>p_plus=True</code> baseline for this report.</div>

<div class="finding"><strong>2. p_plus ablation — overall effect is small but consistently negative on clean data.</strong>
Across all shared combos, p_plus=False reduces clean accuracy by <strong>{avg_clean_delta_nsp:+.4f}</strong>
on average (NSP true-label) and <strong>{avg_clean_delta_ia:+.4f}</strong> (NSP-IA).
At M-0.25, the delta is <strong>{avg_m25_delta_nsp:+.4f}</strong> (NSP) and <strong>{avg_m25_delta_ia:+.4f}</strong> (NSP-IA).
This confirms that the 20-token bank (p_plus=True, ~10K params/filter) provides modest but consistent
gains over single-token prompts (~256 params/filter) — the extra capacity helps, but not dramatically.
The average rank shift between p_plus=True vs False is {avg_rank_shift:.1f} positions — combo ordering
is largely preserved.</div>

<div class="bad"><strong>3. NSP true-label: accuracy collapses under attack regardless of p_plus.</strong>
p_plus=True:  best clean = {nsp_best_clean_val:.4f} → best M-0.25 = {nsp_best_m25_val:.4f} (drop = {nsp_clean_to_m25:.3f}).
p_plus=False: best clean = {np_best_clean_val:.4f} → best M-0.25 = {np_best_m25_val:.4f} (drop = {np_clean_to_m25:.3f}).
Under M-0.25, the best any combo achieves is ~{nsp_best_m25_val:.4f} — barely above random (1/7 ≈ 0.143).
The p_plus parameter does not rescue robustness.</div>

<div class="warn"><strong>4. NSP-IA: near-flat accuracy across ptb — confirms test leakage (unchanged).</strong>
p_plus=True:  best clean = {ia_best_clean_val:.4f} → best M-0.25 = {ia_best_m25_val:.4f} (drop = {ia_clean_to_m25:.3f}).
p_plus=False: best clean = {ianp_best_clean_val:.4f} → best M-0.25 = {ianp_best_m25_val:.4f} (drop = {ianp_clean_to_m25:.3f}).
The IA pseudo-label expansion (35→~595 from test set) creates apparent robustness via transduction leakage.
p_plus=False makes IA clean accuracy slightly worse — the pseudo-label noise hurts more when prompt
capacity is reduced.</div>

<div class="finding"><strong>5. p_plus=False changes best combo at clean but not at M-0.25.</strong>
NSP clean best: p_plus=True → <strong>{nsp_best_clean_combo}</strong> ({nsp_best_clean_val:.4f}),
p_plus=False → <strong>{np_best_clean_combo}</strong> ({np_best_clean_val:.4f}).
NSP M-0.25 best: p_plus=True → <strong>{nsp_best_m25_combo}</strong> ({nsp_best_m25_val:.4f}),
p_plus=False → <strong>{np_best_m25_combo}</strong> ({np_best_m25_val:.4f}).
</div>

<div class="bad"><strong>6. sim_pt is unstable in both p_plus modes.</strong>
p_plus=True: sim has {sim_nan_count} ptb levels with 0/5 valid seeds. p_plus=False: {sim_nan_count_np} NaN levels.
The instability persists regardless of token capacity — it is a fundamental issue with the cosine-similarity
detection threshold, not a parameterization problem.</div>

<div class="finding"><strong>7. Single-filter ranking is consistent across p_plus modes.</strong>
p_plus=True:  {nsp_rank_str}
p_plus=False: {np_rank_str}
Both modes agree that {nsp_best_single_name} is the best single filter on clean data. The order of
the 5 single filters is largely preserved, confirming the relative filter quality is a property
of the detection method, not the prompt token capacity.</div>

<div class="finding"><strong>8. nsp_pt and focusedcleaner_pt remain competitive new additions.</strong>
Both new filters hold their own across p_plus modes:
nsp: {nsp_clean_val:.4f} (pp=True) vs {nsp_clean_np:.4f} (pp=False);
fc:  {fc_clean_val:.4f} (pp=True) vs {fc_clean_np:.4f} (pp=False);
degree: {deg_clean_val:.4f} (pp=True) vs {deg_clean_np:.4f} (pp=False).
At M-0.25, fc is the best single filter in both modes: {fc_m25_nsp:.4f} (pp=True) / {fc_m25_np:.4f} (pp=False).</div>

<div class="warn"><strong>9. IA-nopplus 4/5-combo data incomplete — excluded from analysis.</strong>
The NSP-IA + p_plus=False experiments for 4-filter and 5-filter combos had incomplete seed data
(0-3 valid seeds out of 5 at most ptb levels). These are excluded from all tabs and analysis.
Only 1/2/3-filter combos (25 groups × 6 ptb = 150 experiments) are shown for this variant.</div>

<hr>
<h3>Data Source Summary</h3>
<table class="data-table">
<thead><tr><th>Source</th><th>Prompt Type</th><th>p_plus</th><th>Combos</th><th>Total Runs</th><th>Note</th></tr></thead>
<tbody>
<tr><td><code>logs/5filter_combos/</code></td><td>RobustPrompt-T-NSP</td><td>True</td><td>31 (1-5F)</td><td>186</td><td>Baseline (corrected)</td></tr>
<tr><td><code>logs/5filter_combos_nopplus/</code></td><td>RobustPrompt-T-NSP</td><td>False</td><td>31 (1-5F)</td><td>186</td><td>NEW — single-token prompts</td></tr>
<tr><td><code>logs/5filter_combos_ia/</code></td><td>RobustPrompt-T-NSP-IA</td><td>True</td><td>31 (1-5F)</td><td>186</td><td>IA pseudo-label baseline</td></tr>
<tr><td><code>logs/5filter_combos_ia_nopplus/</code></td><td>RobustPrompt-T-NSP-IA</td><td>False</td><td>25 (1-3F)</td><td>150</td><td>NEW — 4/5F incomplete, excluded</td></tr>
</tbody></table>

<hr>
<h3>Filter Activation Summary</h3>
<p class="note">How each defense tip is activated/detected in <code>RobustPrompt_T_NSP.add_muti_pt()</code>:</p>
<table class="data-table">
<thead><tr><th>Tip Key</th><th>Detection Method</th><th>Threshold Meaning</th><th>Disable Value</th><th>Source</th></tr></thead>
<tbody>
<tr><td><code>sim_pt</code></td><td>Avg neighbor cosine similarity &le; threshold</td><td>Lower = stricter (fewer nodes flagged)</td><td>-1.0</td><td>RobustPrompt_T_NSP.py</td></tr>
<tr><td><code>degree_pt</code></td><td>Node degree &le; threshold</td><td>Lower = stricter</td><td>-1</td><td>RobustPrompt_T_NSP.py</td></tr>
<tr><td><code>out_detect_pt</code></td><td>Edge cosine similarity &le; threshold → both endpoints flagged</td><td>Lower = stricter</td><td>-1.0</td><td>RobustPrompt_T_NSP.py</td></tr>
<tr><td><code>nsp_pt</code></td><td>Neighbor-embedding (A<sup>order</sup>·X) cosine &le; threshold → both endpoints flagged</td><td>Lower = stricter</td><td>-1.0</td><td>filters/nsp_filter.py</td></tr>
<tr><td><code>focusedcleaner_pt</code></td><td>LinkPrediction MLP → low-prob edges → endpoint nodes flagged</td><td>gmean auto-threshold (threshold param unused in default mode)</td><td>-1.0 (now properly disables via pt_dict filtering)</td><td>filters/focusedcleaner_lp_filter.py</td></tr>
<tr><td><code>other_pt</code></td><td>All nodes not covered by any defense tip</td><td>'all' or 'random-X'</td><td>N/A (always active)</td><td>RobustPrompt_T_NSP.py</td></tr>
</tbody></table>
'''

# Format strings
nsp_rank_str = ' &gt; '.join('{}={:.4f}'.format(c, v) for c, v in nsp_singles)
np_rank_str = ' &gt; '.join('{}={:.4f}'.format(c, v) for c, v in np_singles)
ia_rank_str = ' &gt; '.join('{}={:.4f}'.format(c, v) for c, v in ia_singles)
ianp_rank_str = ' &gt; '.join('{}={:.4f}'.format(c, v) for c, v in ianp_singles)

nsp_best_clean_combo = nsp_best_clean[0] or '?'
nsp_best_clean_val = nsp_best_clean[1] if nsp_best_clean[0] else 0
nsp_best_m25_combo = nsp_best_m25[0] or '?'
nsp_best_m25_val = nsp_best_m25[1] if nsp_best_m25[0] else 0
np_best_clean_combo = np_best_clean[0] or '?'
np_best_clean_val = np_best_clean[1] if np_best_clean[0] else 0
np_best_m25_combo = np_best_m25[0] or '?'
np_best_m25_val = np_best_m25[1] if np_best_m25[0] else 0
ia_best_clean_combo = ia_best_clean[0] or '?'
ia_best_clean_val = ia_best_clean[1] if ia_best_clean[0] else 0
ia_best_m25_combo = ia_best_m25[0] or '?'
ia_best_m25_val = ia_best_m25[1] if ia_best_m25[0] else 0
ianp_best_clean_combo = ianp_best_clean[0] or '?'
ianp_best_clean_val = ianp_best_clean[1] if ianp_best_clean[0] else 0
ianp_best_m25_combo = ianp_best_m25[0] or '?'
ianp_best_m25_val = ianp_best_m25[1] if ianp_best_m25[0] else 0

analysis_html = analysis_html.format(
    avg_clean_delta_nsp=avg_clean_delta_nsp,
    avg_clean_delta_ia=avg_clean_delta_ia,
    avg_m25_delta_nsp=avg_m25_delta_nsp,
    avg_m25_delta_ia=avg_m25_delta_ia,
    avg_rank_shift=avg_rank_shift,
    nsp_best_clean_val=nsp_best_clean_val,
    nsp_best_m25_val=nsp_best_m25_val,
    nsp_clean_to_m25=nsp_clean_to_m25,
    np_best_clean_val=np_best_clean_val,
    np_best_m25_val=np_best_m25_val,
    np_clean_to_m25=np_clean_to_m25,
    ia_best_clean_val=ia_best_clean_val,
    ia_best_m25_val=ia_best_m25_val,
    ia_clean_to_m25=ia_clean_to_m25,
    ianp_best_clean_val=ianp_best_clean_val,
    ianp_best_m25_val=ianp_best_m25_val,
    ianp_clean_to_m25=ianp_clean_to_m25,
    nsp_best_clean_combo=nsp_best_clean_combo,
    np_best_clean_combo=np_best_clean_combo,
    nsp_best_m25_combo=nsp_best_m25_combo,
    np_best_m25_combo=np_best_m25_combo,
    sim_nan_count=sim_nan_count,
    sim_nan_count_np=sim_nan_count_np,
    nsp_rank_str=nsp_rank_str,
    np_rank_str=np_rank_str,
    nsp_best_single_name=nsp_best_single_name,
    nsp_clean_val=nsp_clean_val,
    nsp_clean_np=nsp_clean_np,
    fc_clean_val=fc_clean_val,
    fc_clean_np=fc_clean_np,
    deg_clean_val=deg_clean_val,
    deg_clean_np=deg_clean_np,
    fc_m25_nsp=fc_m25_nsp,
    fc_m25_np=fc_m25_np,
)

html += '<div id="tab-analysis" class="tab-panel">\n'
html += analysis_html
html += '</div><!-- end tab-analysis -->\n'

html += '''
</div><!-- end container -->
</body>
</html>'''

out_path = 'advanced_report/5filter_combo_report.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print('Done: {} chars -> {}'.format(len(html), out_path))

# Also dump JSON for programmatic use
with open('advanced_report/5filter_combo_data.json', 'w') as f:
    json.dump({
        'nsp_pp_true': results_nsp,
        'nsp_ia_pp_true': results_ia,
        'nsp_pp_false': results_nsp_np,
        'nsp_ia_pp_false': results_ia_np,
    }, f, indent=2, default=str)
print('JSON data saved to advanced_report/5filter_combo_data.json')
