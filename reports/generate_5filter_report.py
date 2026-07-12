#!/usr/bin/env python3
"""5-filter combo HTML report — clean style, matching meeting15_part1_combo_v2_safe.html."""
import os, re, numpy as np
from collections import defaultdict

log_dir = 'logs/5filter_combos'
results = defaultdict(dict)

for fname in sorted(os.listdir(log_dir)):
    if not fname.endswith('.log'): continue
    m = re.match(r'(.+)_ptb([\d.]+)\.log$', fname)
    if not m: continue
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
        'mean': np.mean(v) if v else None,
        'std': np.std(v) if len(v) > 1 else 0.0,
        'n': len(v),
        'seeds_str': ', '.join('{:.4f}'.format(x) for x in sorted(v))
    }

ptbs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

best_per_col = {}
for p in ptbs:
    vals = [(c, r['mean']) for c in results if (r := results[c].get(p)) and r['mean'] is not None]
    best_per_col[p] = max(v for _, v in vals)

def combo_group(c):
    if c == 'all5': return 5
    return len(c.split('+'))

def sort_key(c):
    return (combo_group(c), c)

section_css = {1: 'section-single', 2: 'section-two', 3: 'section-three', 4: 'section-four', 5: 'section-five'}
seed_notes = []
rows_html = ''

for combo in sorted(results.keys(), key=sort_key):
    n = combo_group(combo)
    css = section_css.get(n, '')
    rows_html += '<tr class="{}"><td>{}</td>'.format(css, combo)
    seed_parts = []
    for p in ptbs:
        r = results[combo].get(p, {})
        m = r.get('mean')
        s = r.get('std', 0)
        nv = r.get('n', 0)
        sd = r.get('seeds_str', '')
        is_best = m is not None and abs(m - best_per_col[p]) < 0.0005
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

seed_html = ''
for combo, detail in seed_notes:
    seed_html += '<div class="seed-detail"><strong>{}:</strong> {}</div>\n'.format(combo, detail)

best_rows = ''
for p in ptbs:
    ranked = [(c, r['mean']) for c in results if (r := results[c].get(p)) and r['mean'] is not None]
    ranked.sort(key=lambda x: -x[1])
    if len(ranked) >= 2:
        best_rows += '<tr><td>{:.2f}</td><td><strong>{}</strong></td><td><strong>{:.4f}</strong></td><td>{} ({:.4f})</td></tr>\n'.format(
            p, ranked[0][0], ranked[0][1], ranked[1][0], ranked[1][1])

html = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>5-Filter Combo Experiment Report</title>
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
  hr {{ border: none; border-top: 1px solid #eee; margin: 20px 0; }}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>5-Filter Combo Experiment Report</h1>
  <div class="meta">
    Generated: 2026-07-12 &nbsp;|&nbsp;
    Peak BB: permE-maskN lr=0.001 r=0.3 seed=1 &nbsp;|&nbsp;
    Code: RobustPrompt-T-NSP (our modified code) &nbsp;|&nbsp;
    Values: mean &plusmn; std across all 5 seeds (no min removal)
  </div>
</header>

<h2>1. Accuracy Across All Filter Combinations</h2>
<p class="note">
  <span style="background:#f0f4ff">&nbsp;Blue&nbsp;</span> = 2-filter,
  <span style="background:#e8f0fe">&nbsp;Lt blue&nbsp;</span> = 3-filter,
  <span style="background:#dbeafe">&nbsp;Dk blue&nbsp;</span> = 4-filter,
  <span style="background:#c7d9f7;font-weight:600">&nbsp;Bold&nbsp;</span> = 5-filter.
  <span style="color:#059669;font-weight:700">Green bold</span> = best in column.
  <code>--</code> = all seeds NaN. <code>*</code> = fewer than 5 valid seeds.
</p>
<table class="data-table">
<thead><tr><th>Combo</th>'''

for p in ptbs:
    html += '<th>{:.2f}</th>'.format(p)
html += '</tr></thead><tbody>\n'
html += rows_html
html += '</tbody></table>\n'

html += '<details>\n<summary>Per-combo seed details (click to expand)</summary>\n'
html += seed_html
html += '</details>\n<hr>\n'

html += '<h3>Best Per Perturbation Level</h3>\n'
html += '<table class="data-table"><thead><tr><th>ptb</th><th>Best Combo</th><th>Accuracy</th><th>Runner-up</th></tr></thead><tbody>\n'
html += best_rows
html += '</tbody></table>\n<hr>\n'

html += '''
<h2>2. Key Findings</h2>

<div class="good"><strong>1. ood is the best single filter.</strong> Clean 0.6062, ptb=0.05 at 0.2431. Leads at low perturbation. At ptb=0.15+ no single filter dominates.</div>

<div class="bad"><strong>2. sim degrades performance and causes NaN instability.</strong> sim alone clean=0.5236 (lowest), with NaN seeds at ptb=0.05 and 0.10. Adding sim to <em>any</em> combo reduces clean accuracy. <strong>Recommendation: exclude sim from future experiments.</strong></div>

<div class="finding"><strong>3. Multi-filter combos provide negligible gains.</strong> Best single filter (ood=0.6062) vs best 3-filter (ood+nsp+fc=0.6141) = +0.8 percentage points. all5 (0.6047) is <em>worse</em> than ood alone.</div>

<div class="finding"><strong>4. fc and nsp are on par with degree.</strong> fc=0.5963, nsp=0.5939, degree=0.5980. Near-identical single-filter performance. Their value is enabling combos without the sim penalty.</div>

<div class="good"><strong>5. degree-based combos lead at high ptb (0.20-0.25).</strong> degree=0.1308, deg+fc=0.1300. degree structural simplicity provides robustness under heavy attack.</div>

<div class="finding"><strong>6. Filter architecture note.</strong> The <code>focusedcleaner_pt</code> defense tip (fc) calls <code>FocusedCleanerLPFilter</code> from <code>focusedcleaner_lp_filter.py</code> — the collaborator\'s LP logic is used as a node-level defense prompt. The <code>--filter_mode focusedcleaner_lp</code> edge filter (not used here; <code>--filter_mode original</code>) is a separate path available in filter_factory.py. Both share the same underlying LP logic from the same file.</div>

</div>
</body>
</html>'''

with open('advanced_report/5filter_combo_report.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Done: {} chars'.format(len(html)))
