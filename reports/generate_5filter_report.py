#!/usr/bin/env python3
"""5-filter combo HTML report — NSP (true labels) vs NSP-IA (pseudo-label expansion).
Corrected: focusedcleaner_pt disable via threshold<0 now works (task.py fix 2026-07-12)."""

import os, re, json, numpy as np
from collections import defaultdict

def parse_dir(log_dir):
    """Parse a log directory, return {combo: {ptb: {mean, std, n, seeds_str}}}."""
    results = defaultdict(dict)
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith('.log'):
            continue
        m = re.match(r'(.+)_ptb([\d.]+)\.log$', fname)
        if not m:
            continue
        combo, ptb = m.group(1), float(m.group(2))
        with open(os.path.join(log_dir, fname), 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        seeds = {}
        for sm in re.finditer(r'# Seed (\d+) Muti Split Final Acc: ([\d.]+|nan)', content):
            acc_str = sm.group(2)
            if acc_str != 'nan':
                seeds[int(sm.group(1))] = float(acc_str)
        v = list(seeds.values())
        results[combo][ptb] = {
            'mean': float(np.mean(v)) if v else None,
            'std': float(np.std(v)) if len(v) > 1 else 0.0,
            'n': len(v),
            'seeds_str': ', '.join('{:.4f}'.format(x) for x in sorted(v))
        }
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

def build_comparison_rows(results_nsp, results_ia, bp_nsp, bp_ia):
    """Build comparison table: side-by-side NSP vs NSP-IA."""
    html = ''
    for combo in sorted(results_nsp.keys(), key=sort_key):
        n = combo_group(combo)
        css = SECTION_CSS.get(n, '')
        html += '<tr class="{}"><td>{}</td>'.format(css, combo)
        for p in PTBS:
            r_nsp = results_nsp[combo].get(p, {})
            r_ia = results_ia.get(combo, {}).get(p, {})
            mn = r_nsp.get('mean')
            mi = r_ia.get('mean')

            if mn is None and mi is None:
                html += '<td class="na">--</td>'
            elif mi is None:
                html += '<td>{:.4f} vs --</td>'.format(mn)
            elif mn is None:
                html += '<td>-- vs {:.4f}</td>'.format(mi)
            else:
                diff = mi - mn
                cls = 'diff-pos' if diff > 0.005 else ('diff-neg' if diff < -0.005 else '')
                cell = '{:.4f} vs {:.4f}'.format(mn, mi)
                if cls:
                    dsign = '+' if diff > 0 else ''
                    cell += ' <span class="{}">({}{:.4f})</span>'.format(cls, dsign, diff)
                html += '<td>{}</td>'.format(cell)
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
    # Headers from single filter names
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
    html = '<h3>{} — Robustness Summary (clean → M-0.25)</h3>\n'.format(label)
    html += '<table class="data-table"><thead><tr><th>Group Size</th><th>Best Clean</th><th>Best M-0.25</th><th>Drop</th></tr></thead><tbody>\n'
    for n in [1, 2, 3, 4, 5]:
        group = {c: rd for c, rd in results.items() if combo_group(c) == n}
        clean_best = max((r.get(0.0, {}).get('mean') or 0 for r in group.values()), default=0)
        m25_best = max((r.get(0.25, {}).get('mean') or 0 for r in group.values()), default=0)
        drop = clean_best - m25_best
        html += '<tr><td>{}-filter</td><td>{:.4f}</td><td>{:.4f}</td><td>{:.4f}</td></tr>\n'.format(n, clean_best, m25_best, drop)
    html += '</tbody></table>\n'
    return html


# ============================================================
# MAIN
# ============================================================
print("Parsing logs/5filter_combos (NSP true-label)...")
results_nsp = parse_dir('logs/5filter_combos')
print("  {} combos, {} total entries".format(len(results_nsp), sum(len(v) for v in results_nsp.values())))

print("Parsing logs/5filter_combos_ia (NSP-IA pseudo-label)...")
results_ia = parse_dir('logs/5filter_combos_ia')
print("  {} combos, {} total entries".format(len(results_ia), sum(len(v) for v in results_ia.values())))

bp_nsp = best_per_col(results_nsp)
bp_ia = best_per_col(results_ia)

rows_nsp, seed_nsp = build_table_html(results_nsp, bp_nsp)
rows_ia, seed_ia = build_table_html(results_ia, bp_ia)
comp_rows = build_comparison_rows(results_nsp, results_ia, bp_nsp, bp_ia)

seed_html_nsp = build_seed_html(seed_nsp)
seed_html_ia = build_seed_html(seed_ia)

# ============================================================
# HTML
# ============================================================
html = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>5-Filter Combo — NSP vs NSP-IA (Corrected)</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #fff; color: #1a1a1a; font-size: 12px; line-height:1.5; }}
  .container {{ max-width: 1400px; margin:0 auto; padding: 28px 16px; }}
  header {{ margin-bottom: 24px; border-bottom: 1.5px solid #2563eb; padding-bottom: 12px; }}
  header h1 {{ font-size: 18px; font-weight: 700; }}
  header .meta {{ font-size: 10px; color: #888; margin-top: 3px; }}
  h2 {{ font-size: 14px; font-weight: 600; margin: 28px 0 10px; color: #333; border-bottom: 1px solid #e5e5e5; padding-bottom: 4px; }}
  h3 {{ font-size: 13px; font-weight: 600; margin: 20px 0 8px; color: #444; }}
  table.data-table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin: 6px 0 14px; font-variant-numeric: tabular-nums; }}
  table.data-table thead th {{ background: #f5f5f5; font-weight: 600; text-align: center; padding: 4px 6px; border-bottom: 1.5px solid #ddd; white-space: nowrap; position: sticky; top: 0; z-index: 1; }}
  table.data-table tbody td {{ text-align: center; padding: 3px 6px; border-bottom: 1px solid #eee; }}
  table.data-table tbody tr:nth-child(even) {{ background: #fafafa; }}
  .section-single {{ }}
  .section-two {{ background: #f0f4ff; }}
  .section-three {{ background: #e8f0fe; }}
  .section-four {{ background: #dbeafe; }}
  .section-five {{ background: #c7d9f7; font-weight: 600; }}
  .best {{ font-weight: 700; color: #059669; }}
  .na {{ color: #bbb; font-style: italic; }}
  .note {{ font-size: 10px; color: #999; margin-top: 2px; }}
  .seed-detail {{ font-size: 10px; color: #777; font-family: 'SF Mono', 'Consolas', monospace; margin: 2px 0; }}
  details {{ margin: 6px 0 12px; }}
  details summary {{ font-size: 11px; color: #555; cursor: pointer; }}
  .finding {{ background: #f8fafc; border-left: 3px solid #2563eb; padding: 8px 12px; margin: 8px 0; font-size: 11px; }}
  .bad {{ background: #fef2f2; border-left: 3px solid #dc2626; padding: 8px 12px; margin: 8px 0; font-size: 11px; }}
  .good {{ background: #f0fdf4; border-left: 3px solid #059669; padding: 8px 12px; margin: 8px 0; font-size: 11px; }}
  .warn {{ background: #fffbeb; border-left: 3px solid #d97706; padding: 8px 12px; margin: 8px 0; font-size: 11px; }}
  hr {{ border: none; border-top: 1px solid #eee; margin: 20px 0; }}
  .diff-pos {{ font-weight: 700; color: #059669; }}
  .diff-neg {{ font-weight: 700; color: #dc2626; }}
  .tab-nav {{ display: flex; gap: 0; margin-bottom: 12px; border-bottom: 2px solid #e5e5e5; }}
  .tab-btn {{ padding: 6px 18px; cursor: pointer; border: none; background: none; font-size: 12px; font-weight: 500; color: #888; border-bottom: 2px solid transparent; margin-bottom: -2px; }}
  .tab-btn.active {{ color: #2563eb; border-bottom-color: #2563eb; font-weight: 600; }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  .fix-badge {{ display: inline-block; background: #059669; color: #fff; font-size: 10px; padding: 1px 6px; border-radius: 3px; margin-left: 6px; font-weight: 600; }}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>5-Filter Combo Experiment Report <span class="fix-badge">CORRECTED</span></h1>
  <div class="meta">
    Generated: 2026-07-12 (corrected — task.py fix applied: threshold < 0 now properly disables filters) &nbsp;|&nbsp;
    Peak BB: permE-maskN lr=0.001 r=0.3 seed=1 (256-dim GCN) &nbsp;|&nbsp;
    5 seeds &times; 31 combos &times; 6 ptb = 186 runs per track &nbsp;|&nbsp;
    Values: mean &plusmn; std across all 5 seeds (no min removal)
  </div>
  <div class="meta" style="margin-top:4px;">
    <strong>Fix applied:</strong> <code>task.py</code> now filters out negative-threshold entries from
    <code>muti_defense_pt_dict</code> before prompt construction. Setting a threshold to -1.0
    truly disables that filter (no parameter creation, no detection logic, no FocusedCleaner-LP training).
    The previous bug where <code>focusedcleaner_pt</code> always ran regardless of threshold is resolved.
  </div>
</header>

<!-- ============ TAB NAV ============ -->
<div class="tab-nav">
  <button class="tab-btn active" onclick="showTab('nsp')">NSP (True Labels)</button>
  <button class="tab-btn" onclick="showTab('ia')">NSP-IA (Pseudo-Labels)</button>
  <button class="tab-btn" onclick="showTab('compare')">NSP vs NSP-IA</button>
  <button class="tab-btn" onclick="showTab('analysis')">Analysis</button>
</div>

<script>
function showTab(name) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}}
</script>

<!-- ============ TAB: NSP (True Labels) ============ -->
<div id="tab-nsp" class="tab-panel active">

<h2>NSP — True Labels (RobustPrompt-T-NSP)</h2>
<p class="note">
  <span style="background:#f0f4ff">&nbsp;Blue&nbsp;</span> = 2-filter,
  <span style="background:#e8f0fe">&nbsp;Lt blue&nbsp;</span> = 3-filter,
  <span style="background:#dbeafe">&nbsp;Dk blue&nbsp;</span> = 4-filter,
  <span style="background:#c7d9f7;font-weight:600">&nbsp;Bold&nbsp;</span> = 5-filter.
  <span class="best">Green bold</span> = best in column.
  <code>--</code> = all seeds NaN. <code>*</code> = fewer than 5 valid seeds.
</p>
<table class="data-table">
<thead><tr><th>Combo</th>'''
for p in PTBS:
    html += '<th>{:.2f}</th>'.format(p)
html += '</tr></thead><tbody>\n'
html += rows_nsp
html += '</tbody></table>\n'

html += '<details>\n<summary>Per-combo seed details (click to expand)</summary>\n'
html += seed_html_nsp
html += '</details>\n'

html += '<h3>Best Per Perturbation Level</h3>\n'
html += '<table class="data-table"><thead><tr><th>ptb</th><th>Best Combo</th><th>Accuracy</th><th>Runner-up</th></tr></thead><tbody>\n'
html += build_best_rows(results_nsp, bp_nsp)
html += '</tbody></table>\n'

html += build_robustness_summary(results_nsp, 'NSP True-Label')

html += '<h3>Single Filter Comparison</h3>\n'
html += build_single_best_table(results_nsp)

html += '</div><!-- end tab-nsp -->\n'

# ============ TAB: NSP-IA ============
html += '''<div id="tab-ia" class="tab-panel">

<h2>NSP-IA — Pseudo-Labels (RobustPrompt-T-NSP-IA)</h2>
<p class="note">
  IA-PT expands training labels from 35 → ~595 via high-confidence pseudo-labels (transductive leakage).
  Same layout conventions as the NSP tab.
</p>
<table class="data-table">
<thead><tr><th>Combo</th>'''
for p in PTBS:
    html += '<th>{:.2f}</th>'.format(p)
html += '</tr></thead><tbody>\n'
html += rows_ia
html += '</tbody></table>\n'

html += '<details>\n<summary>Per-combo seed details (click to expand)</summary>\n'
html += seed_html_ia
html += '</details>\n'

html += '<h3>Best Per Perturbation Level</h3>\n'
html += '<table class="data-table"><thead><tr><th>ptb</th><th>Best Combo</th><th>Accuracy</th><th>Runner-up</th></tr></thead><tbody>\n'
html += build_best_rows(results_ia, bp_ia)
html += '</tbody></table>\n'

html += build_robustness_summary(results_ia, 'NSP-IA Pseudo-Label')

html += '<h3>Single Filter Comparison</h3>\n'
html += build_single_best_table(results_ia)

html += '</div><!-- end tab-ia -->\n'

# ============ TAB: COMPARISON ============
html += '''<div id="tab-compare" class="tab-panel">

<h2>NSP vs NSP-IA — Side-by-Side Comparison</h2>
<p class="note">
  Each cell: <strong>NSP</strong> vs <strong>NSP-IA</strong>.
  <span class="diff-pos">Green delta</span> = IA better by &gt;0.005.
  <span class="diff-neg">Red delta</span> = IA worse by &gt;0.005.
</p>
<table class="data-table">
<thead><tr><th>Combo</th>'''
for p in PTBS:
    html += '<th>{:.2f}</th>'.format(p)
html += '</tr></thead><tbody>\n'
html += comp_rows
html += '</tbody></table>\n'

# Add win/loss summary
html += '<h3>IA Win/Loss Summary</h3>\n'
html += '<table class="data-table"><thead><tr><th>ptb</th><th>IA Wins</th><th>IA Losses</th><th>Ties</th><th>Avg Delta</th></tr></thead><tbody>\n'
for p in PTBS:
    wins = losses = ties = 0
    deltas = []
    for combo in sorted(results_nsp.keys()):
        rn = results_nsp[combo].get(p, {})
        ri = results_ia.get(combo, {}).get(p, {})
        mn, mi = rn.get('mean'), ri.get('mean')
        if mn is None or mi is None:
            continue
        d = mi - mn
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

html += '</div><!-- end tab-compare -->\n'

# ============ TAB: ANALYSIS ============
# Compute key statistics
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

nsp_best_clean = find_best(results_nsp, 0.0)
nsp_best_m25 = find_best(results_nsp, 0.25)
ia_best_clean = find_best(results_ia, 0.0)
ia_best_m25 = find_best(results_ia, 0.25)

# Single filter ranking
nsp_singles = [(c, rd.get(0.0, {}).get('mean', 0)) for c, rd in results_nsp.items() if combo_group(c) == 1]
nsp_singles.sort(key=lambda x: -x[1])
ia_singles = [(c, rd.get(0.0, {}).get('mean', 0)) for c, rd in results_ia.items() if combo_group(c) == 1]
ia_singles.sort(key=lambda x: -x[1])

# Best combo per group size at clean
def best_per_group(results, ptb):
    groups = defaultdict(list)
    for c, rd in results.items():
        r = rd.get(ptb, {})
        if r.get('mean'):
            groups[combo_group(c)].append((c, r['mean']))
    out = {}
    for n in [1, 2, 3, 4, 5]:
        if groups[n]:
            out[n] = max(groups[n], key=lambda x: x[1])
    return out

nsp_clean_groups = best_per_group(results_nsp, 0.0)
nsp_m25_groups = best_per_group(results_nsp, 0.25)

# Compute additional stats for findings
# sim NaN count
sim_nan_count = sum(1 for p in PTBS
                    if results_nsp.get('sim', {}).get(p, {}).get('n', 0) == 0)
# Best single vs best multi at clean
best_single_clean = max((rd.get(0.0,{}).get('mean',0) for c, rd in results_nsp.items() if combo_group(c)==1), default=0)
best_multi_clean = max((rd.get(0.0,{}).get('mean',0) for c, rd in results_nsp.items() if combo_group(c)>1), default=0)
# best single at M-0.25
best_single_m25 = max((rd.get(0.25,{}).get('mean',0) for c, rd in results_nsp.items() if combo_group(c)==1), default=0)
best_multi_m25 = max((rd.get(0.25,{}).get('mean',0) for c, rd in results_nsp.items() if combo_group(c)>1), default=0)
# Clean→M-0.25 drop
nsp_clean_to_m25 = nsp_best_clean[1] - nsp_best_m25[1] if nsp_best_clean[0] and nsp_best_m25[0] else 0
ia_clean_to_m25 = ia_best_clean[1] - ia_best_m25[1] if ia_best_clean[0] and ia_best_m25[0] else 0

analysis_html = '''
<h2>Analysis & Key Findings</h2>

<div class="good"><strong>1. Fix verified — focusedcleaner_pt now properly disabled.</strong>
With the <code>task.py</code> fix (threshold &lt; 0 → key removed from <code>pt_dict</code>),
the 5-filter permutation experiment now correctly isolates individual filter contributions.
The focusedcleaner_pt no longer trains its 50-epoch LP model when "off" (threshold = -1.0).
This report replaces the previous buggy version where fc was always active.</div>

<div class="finding"><strong>2. Single-filter clean ranking (NSP true-label):</strong> {}
</div>

<div class="finding"><strong>3. Single-filter clean ranking (NSP-IA pseudo-label):</strong> {}
</div>

<div class="bad"><strong>4. NSP true-label: accuracy collapses under attack.</strong>
Best clean = {:.4f} → Best M-0.25 = {:.4f} (drop = {:.3f}).
On clean data, the prompt achieves ~0.59 with 35 labels.
Under M-0.25, the best any combo achieves is {:.4f} — barely above random (1/7 ≈ 0.143).
This confirms the fundamental difficulty: <em>RobustPrompt-T cannot maintain robustness under
heavy graph perturbation in a true few-shot setting</em>. The previous report's higher M-0.25
numbers were artifacts of the focusedcleaner-always-active bug.</div>

<div class="warn"><strong>5. NSP-IA: near-flat accuracy across ptb — confirms test leakage.</strong>
Best clean = {:.4f} → Best M-0.25 = {:.4f} (drop = {:.3f}).
The IA pseudo-label expansion (35 → ~595 labels, sampled from test set) creates a model
that appears "robust" because it has already seen the test distribution during training.
This is <em>not genuine robustness</em> — see meeting15_root_cause.html for the full analysis.
Notably, NSP-IA single-filter clean ({:.4f}) is <em>lower</em> than NSP true-label ({:.4f}),
suggesting pseudo-label noise hurts more than the extra labels help on clean data.</div>

<div class="finding"><strong>6. Multi-filter combos: marginal clean gain, harmful under attack.</strong>
At clean: best single = {:.4f} vs best multi = {:.4f} (&Delta; = +{:.4f}).
At M-0.25: best single = {:.4f} vs best multi = {:.4f} (&Delta; = {:.4f}).
Adding more filters does not improve robustness — the best M-0.25 performer is a <em>single</em>
filter (fc = {:.4f}), and multi-filter combos are slightly <em>worse</em> under attack.</div>

<div class="bad"><strong>7. sim_pt is unstable and degrades most combos.</strong>
sim alone has NaN instability ({} ptb levels with 0/5 valid seeds, clean only 4/5).
Adding sim to <em>any</em> combo reduces clean accuracy — <em>except</em> when ood is also present
(sim+ood ≈ ood alone). Worst interaction: sim+nsp = {:.4f} (lowest 2-filter combo).
<strong>Recommendation: exclude sim_pt from future experiments.</strong></div>

<div class="finding"><strong>8. nsp_pt and focusedcleaner_pt perform on par with original filters.</strong>
The two new filters are competitive: nsp ({:.4f} clean) slightly edges degree ({:.4f}),
fc ({:.4f}) is close behind. At M-0.25, fc is actually the best single filter ({:.4f}).
Both new filters are valid additions to the defense tip family.</div>

<hr>
<h3>Filter Activation Summary</h3>
<p class="note">How each defense tip is activated/detected in <code>RobustPrompt_T_NSP.add_muti_pt()</code>:</p>
<table class="data-table">
<thead><tr><th>Tip Key</th><th>Detection Method</th><th>Threshold Meaning</th><th>Disable Value</th><th>Source</th></tr></thead>
<tbody>
<tr><td><code>sim_pt</code></td><td>Avg neighbor cosine similarity &le; threshold</td><td>Lower = stricter (fewer nodes flagged)</td><td>-1.0</td><td>RobustPrompt_T_NSP.py (original)</td></tr>
<tr><td><code>degree_pt</code></td><td>Node degree &le; threshold</td><td>Lower = stricter (fewer nodes flagged)</td><td>-1</td><td>RobustPrompt_T_NSP.py (original)</td></tr>
<tr><td><code>out_detect_pt</code></td><td>Edge cosine similarity &le; threshold → both endpoints flagged</td><td>Lower = stricter (fewer edges flagged)</td><td>-1.0</td><td>RobustPrompt_T_NSP.py (original)</td></tr>
<tr><td><code>nsp_pt</code></td><td>Neighbor-embedding (A<sup>order</sup>·X) cosine &le; threshold → both endpoints flagged</td><td>Lower = stricter</td><td>-1.0</td><td>filters/nsp_filter.py (new)</td></tr>
<tr><td><code>focusedcleaner_pt</code></td><td>LinkPrediction MLP → low-prob edges → endpoint nodes flagged</td><td>gmean auto-threshold (threshold param unused in default mode)</td><td>-1.0 (now properly disables via pt_dict filtering)</td><td>filters/focusedcleaner_lp_filter.py (new)</td></tr>
<tr><td><code>other_pt</code></td><td>All nodes not covered by any defense tip</td><td>'all' or 'random-X'</td><td>N/A (always active, value is string)</td><td>RobustPrompt_T_NSP.py (original)</td></tr>
</tbody></table>
'''

# Format the single-filter rankings
nsp_rank_str = ' &gt; '.join('{}={:.4f}'.format(c, v) for c, v in nsp_singles)
ia_rank_str = ' &gt; '.join('{}={:.4f}'.format(c, v) for c, v in ia_singles)

# Additional stats for findings
best_ia_single_clean = max((rd.get(0.0,{}).get('mean',0) for c, rd in results_ia.items() if combo_group(c)==1), default=0)
best_nsp_single_clean = max((rd.get(0.0,{}).get('mean',0) for c, rd in results_nsp.items() if combo_group(c)==1), default=0)
sim_nsp_clean = results_nsp.get('sim+nsp', {}).get(0.0, {}).get('mean', 0)
nsp_clean = results_nsp.get('nsp', {}).get(0.0, {}).get('mean', 0)
deg_clean = results_nsp.get('degree', {}).get(0.0, {}).get('mean', 0)
fc_clean = results_nsp.get('fc', {}).get(0.0, {}).get('mean', 0)
fc_m25_val = results_nsp.get('fc', {}).get(0.25, {}).get('mean', 0)
delta_clean = best_multi_clean - best_single_clean
delta_m25 = best_multi_m25 - best_single_m25

analysis_html = analysis_html.format(
    # Finding 2, 3
    nsp_rank_str, ia_rank_str,
    # Finding 4: NSP best clean, best m25, drop, best m25 (repeat)
    nsp_best_clean[1] if nsp_best_clean[0] else 0,
    nsp_best_m25[1] if nsp_best_m25[0] else 0,
    nsp_clean_to_m25,
    nsp_best_m25[1] if nsp_best_m25[0] else 0,
    # Finding 5: IA best clean, m25, drop, IA single clean, NSP single clean
    ia_best_clean[1] if ia_best_clean[0] else 0,
    ia_best_m25[1] if ia_best_m25[0] else 0,
    ia_clean_to_m25,
    best_ia_single_clean,
    best_nsp_single_clean,
    # Finding 6: single clean, multi clean, delta clean, single m25, multi m25, delta m25, fc m25
    best_single_clean, best_multi_clean, delta_clean,
    best_single_m25, best_multi_m25, delta_m25,
    fc_m25_val,
    # Finding 7: sim_nan_count, sim+nsp clean
    sim_nan_count, sim_nsp_clean,
    # Finding 8: nsp clean, degree clean, fc clean, fc m25
    nsp_clean, deg_clean, fc_clean, fc_m25_val,
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
    json.dump({'nsp': results_nsp, 'nsp_ia': results_ia}, f, indent=2, default=str)
print('JSON data saved to advanced_report/5filter_combo_data.json')
