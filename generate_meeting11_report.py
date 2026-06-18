"""Generate Meeting 11 PDF report from experiment logs."""
import os
import re
import glob
import numpy as np
from fpdf import FPDF

LOG_DIR = "logs/RobustPrompt-T"

# ── Data extraction ──────────────────────────────────────────────

def extract_accuracies(log_path):
    """Parse a log file, return list of per-seed accuracies."""
    accs = []
    with open(log_path, "r") as f:
        for line in f:
            m = re.search(r"Final True Accuracy:\s*([\d.]+)", line)
            if m:
                accs.append(float(m.group(1)))
    return accs

def find_log(pattern, ptb):
    """Find log matching a pattern and perturbation level. Return path or None."""
    files = sorted(glob.glob(os.path.join(LOG_DIR, f"{pattern}_*{ptb}_*.log")))
    return files[0] if files else None

def get_result(pattern, ptb):
    """Return (accs_list, mean, std) for a pattern+ptb combo, or (None, None, None)."""
    path = find_log(pattern, ptb)
    if path is None:
        return None, None, None
    accs = extract_accuracies(path)
    if not accs:
        return None, None, None
    # MyTask.py drops the lowest seed, then averages. We replicate that.
    if len(accs) >= 2:
        accs_sorted = sorted(accs)
        # Drop min (index 0)
        trimmed = accs_sorted[1:]
        mean_val = np.mean(trimmed)
        std_val = np.std(trimmed, ddof=1)
        return accs, mean_val, std_val
    return accs, np.mean(accs), 0.0

# ── Build report data ────────────────────────────────────────────

def build_report():
    GPPT = {0.0: 0.4350, 0.05: 0.2790, 0.1: 0.0700, 0.15: 0.0740, 0.2: 0.0280, 0.25: 0.0350}
    ptbs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

    # Phase 3a: sim_pt
    sim_results = []
    for sim in [0.2, 0.3, 0.4, 0.5, 0.6]:
        row = {"param": f"sim={sim}"}
        for ptb in ptbs:
            accs, mean, std = get_result(f"ft_sim{sim}", ptb)
            row[ptb] = (accs, mean, std)
        sim_results.append(row)

    # Phase 3b: degree_pt
    deg_results = []
    for deg in [1, 2, 3, 5]:
        row = {"param": f"deg={deg}"}
        for ptb in ptbs:
            accs, mean, std = get_result(f"ft_deg{deg}", ptb)
            row[ptb] = (accs, mean, std)
        deg_results.append(row)

    # Phase 3c: out_detect_pt
    ood_results = []
    for ood in [0.3, 0.4, 0.5, 0.6, 0.7]:
        row = {"param": f"ood={ood}"}
        for ptb in ptbs:
            accs, mean, std = get_result(f"ft_ood{ood}", ptb)
            row[ptb] = (accs, mean, std)
        ood_results.append(row)

    # Phase 3d: Combo A & B
    combo_results = {}
    for combo_name, combo_label, pattern in [
        ("A", "Combo A: sim=0.6, deg=1, ood=0.4 (clean-optimal)", "ft_comboA"),
        ("B", "Combo B: sim=0.3, deg=3, ood=0.4 (attacked-optimal)", "ft_comboB"),
    ]:
        rows = []
        for ptb in ptbs:
            accs, mean, std = get_result(pattern, ptb)
            rows.append((ptb, accs, mean, std))
        combo_results[combo_name] = {"label": combo_label, "rows": rows}

    return GPPT, ptbs, sim_results, deg_results, ood_results, combo_results

# ── PDF Generation ───────────────────────────────────────────────

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "Meeting 11 - RobustPrompt-T Filtering Tips Tuning Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, "2026-06-04 | GPromptShield Paper-Aligned Code (tau_tune only, no filter_module)", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(41, 65, 122)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(41, 65, 122)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, text, align="L")
        self.ln(1)

    def result_table(self, headers, rows, highlight_col=None):
        """Draw a table with headers and rows. highlight_col marks best value in bold."""
        col_w = 190 / len(headers)
        # Header
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(220, 225, 235)
        for h in headers:
            self.cell(col_w, 6, h, border=1, fill=True, align="C")
        self.ln()
        # Rows
        best_idx = None
        if highlight_col is not None and rows:
            valid = [(i, r[highlight_col]) for i, r in enumerate(rows) if r[highlight_col] is not None]
            if valid:
                best_idx = max(valid, key=lambda x: x[1])[0]
        for i, row in enumerate(rows):
            is_best = (i == best_idx)
            self.set_font("Helvetica", "B" if is_best else "", 8)
            if is_best:
                self.set_fill_color(200, 255, 200)
            for j, cell in enumerate(row):
                self.cell(col_w, 5.5, str(cell) if cell is not None else "N/A", border=1, fill=is_best, align="C")
            self.ln()
        self.ln(3)

    def key_finding(self, text):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(255, 245, 200)
        self.multi_cell(0, 5.5, f"  {text}", fill=True)
        self.ln(1)


def build_pdf():
    GPPT, ptbs, sim_results, deg_results, ood_results, combo_results = build_report()
    ptb_labels = ["0.0 (clean)", "0.05", "0.1", "0.15", "0.2", "0.25"]

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── Executive Summary ────────────────────────────────────────
    pdf.section_title("1. Executive Summary")

    pdf.body_text(
        "This report presents the results of Filtering Tips threshold tuning for RobustPrompt-T "
        "(GPromptShield) on the Cora dataset. All experiments use the paper-aligned code "
        "(tau_tune only, no filter_module edge pruning) with fixed training parameters: "
        "no_attention, p_plus=True, prompt_lr=0.01, pt_threshold=0.5, weight_mse=0.1, weight_kl=0.3."
    )
    pdf.body_text(
        "Method: Isolated tuning -- when tuning one threshold, the other two are set to impossible "
        "values (degree=-1, or sim/ood=-1.0), ensuring each phase measures the independent effect "
        "of exactly one defense prompt type."
    )

    # Best single-defense summary
    pdf.sub_title("Best Single-Defense Results vs GPPT Baseline")
    best_headers = ["Metric", "Clean (0.0)", "Attacked (0.05)", "GPPT Clean", "GPPT 0.05"]
    # sim_pt best on clean: sim=0.6
    sim_best_clean = max([r for r in sim_results if r[0.0][1] is not None], key=lambda r: r[0.0][1])
    sim_best_att = max([r for r in sim_results if r[0.05][1] is not None], key=lambda r: r[0.05][1])
    deg_best_clean = max([r for r in deg_results if r[0.0][1] is not None], key=lambda r: r[0.0][1])
    deg_best_att = max([r for r in deg_results if r[0.05][1] is not None], key=lambda r: r[0.05][1])
    ood_best_clean = max([r for r in ood_results if r[0.0][1] is not None], key=lambda r: r[0.0][1])
    ood_best_att = max([r for r in ood_results if r[0.05][1] is not None], key=lambda r: r[0.05][1])

    best_rows = [
        ["sim_pt best clean  (sim=0.6)", f"{sim_best_clean[0.0][1]:.4f}", f"{sim_best_clean[0.05][1]:.4f}" if sim_best_clean[0.05][1] else "N/A", "0.4350", "0.2790"],
        ["sim_pt best 0.05   (sim=0.3)", f"{sim_best_att[0.0][1]:.4f}", f"{sim_best_att[0.05][1]:.4f}", "0.4350", "0.2790"],
        ["degree_pt best clean (deg=1)", f"{deg_best_clean[0.0][1]:.4f}", f"{deg_best_clean[0.05][1]:.4f}" if deg_best_clean[0.05][1] else "N/A", "0.4350", "0.2790"],
        ["degree_pt best 0.05  (deg=3)", f"{deg_best_att[0.0][1]:.4f}", f"{deg_best_att[0.05][1]:.4f}", "0.4350", "0.2790"],
        ["out_detect_pt (ood=0.4)", f"{ood_best_clean[0.0][1]:.4f}", f"{ood_best_clean[0.05][1]:.4f}", "0.4350", "0.2790"],
    ]
    pdf.result_table(best_headers, best_rows)

    pdf.key_finding("KEY: degree_pt (deg=1) achieves 0.4435 clean, surpassing GPPT 0.4350.")
    pdf.key_finding("KEY: out_detect_pt (ood=0.4) achieves 0.4490 clean AND 0.2485 attacked. Only defense where clean-optimal == attacked-optimal.")
    pdf.key_finding("KEY: Clean-optimal thresholds != Attacked-optimal thresholds for sim_pt and degree_pt. This is a critical finding for future tuning strategy.")

    # ── Experimental Setup ───────────────────────────────────────
    pdf.section_title("2. Experimental Setup")
    pdf.body_text(
        "Dataset: Cora (2708 nodes, 5429 edges, 7 classes)\n"
        "Pre-trained model: GraphCL GCN 256-dim (aug1=dropN, aug2=permE, lr=0.01)\n"
        "Attack: Meta_Self (Metattack) at perturbation rates 0.0, 0.05, 0.1, 0.15, 0.2, 0.25\n"
        "Few-shot: 5-shot, split 1\n"
        "Seeds: 1, 2, 3, 4, 5 (lowest seed dropped, mean of remaining 4 reported)\n"
        "Epochs: 200, early stopping patience: 20\n"
        "\n"
        "Fixed Parameters:\n"
        "  no_attention, p_plus=True, prompt_lr=0.01, pt_threshold=0.5\n"
        "  weight_mse=0.1, weight_kl=0.3, weight_constraint=0.2\n"
        "  filter_mode=original\n"
        "\n"
        "Training flow (paper-aligned): add_muti_pt -> GNN1 -> tau_tune edge pruning -> GNN2\n"
        "Eval flow (paper-aligned): add_muti_pt -> GNN (no edge pruning)"
    )

    # ── Phase 3a ─────────────────────────────────────────────────
    pdf.section_title("3. Phase 3a: sim_pt Independent Tuning (degree=-1, ood=-1.0)")
    pdf.body_text(
        "sim_pt is assigned to nodes whose neighbor-averaged cosine similarity <= pt_sim_threshold. "
        "Higher threshold = more nodes get sim_pt. Paper default: 0.4."
    )

    headers = ["Threshold"] + ptb_labels
    rows = []
    for r in sim_results:
        row = [r["param"]]
        for ptb in ptbs:
            _, mean, std = r[ptb]
            row.append(f"{mean:.4f} +/- {std:.4f}" if mean is not None else "N/A")
        rows.append(row)
    pdf.result_table(headers, rows, highlight_col=1)  # highlight best clean

    # Clean best / 0.05 best
    pdf.key_finding(f"Clean-optimal: sim=0.6 ({sim_best_clean[0.0][1]:.4f}), Attacked-optimal: sim=0.3 ({sim_best_att[0.05][1]:.4f}). Direction REVERSES -- clean wants loose sim (many nodes), attacked wants strict sim (few nodes).")

    # ── Phase 3b ─────────────────────────────────────────────────
    pdf.section_title("4. Phase 3b: degree_pt Independent Tuning (sim=-1.0, ood=-1.0)")
    pdf.body_text(
        "degree_pt is assigned to nodes with degree <= pt_degree_threshold. "
        "Higher threshold = more nodes get degree_pt. Paper default: 2."
    )

    rows = []
    for r in deg_results:
        row = [r["param"]]
        for ptb in ptbs:
            _, mean, std = r[ptb]
            row.append(f"{mean:.4f} +/- {std:.4f}" if mean is not None else "N/A")
        rows.append(row)
    pdf.result_table(headers, rows, highlight_col=1)

    pdf.key_finding(f"Clean-optimal: deg=1 ({deg_best_clean[0.0][1]:.4f}) -- only isolated/pendant nodes get degree_pt. Clean 0.4435 BEATS GPPT 0.4350.")
    pdf.key_finding(f"Attacked-optimal: deg=3 ({deg_best_att[0.05][1]:.4f}). On attacked graphs, moderate coverage (deg<=3) outperforms extreme (deg=1 -> 0.2448).")

    # ── Phase 3c ─────────────────────────────────────────────────
    pdf.section_title("5. Phase 3c: out_detect_pt Independent Tuning (sim=-1.0, degree=-1)")
    pdf.body_text(
        "out_detect_pt is assigned to endpoint nodes of edges whose cosine similarity <= pt_out_detect_threshold. "
        "Higher threshold = more edges flagged as OOD = more nodes get out_detect_pt. Paper default: 0.5."
    )

    rows = []
    for r in ood_results:
        row = [r["param"]]
        for ptb in ptbs:
            _, mean, std = r[ptb]
            row.append(f"{mean:.4f} +/- {std:.4f}" if mean is not None else "N/A")
        rows.append(row)
    pdf.result_table(headers, rows, highlight_col=1)

    pdf.key_finding(f"Both clean and attacked optimal at ood=0.4 (clean: {ood_best_clean[0.0][1]:.4f}, attacked: {ood_best_clean[0.05][1]:.4f}). Out of three defense types, only out_detect_pt has consistent optimal threshold across perturbation levels.")
    pdf.key_finding(f"ood=0.4 has the smallest clean->attacked gap (0.4490 -> 0.2485, delta=0.2005), making it the most robust single defense.")

    # ── Phase 3d ─────────────────────────────────────────────────
    pdf.section_title("6. Phase 3d: Combined Verification")
    pdf.body_text(
        "Two combinations tested across all perturbation levels:\n"
        "  Combo A (clean-optimal): sim=0.6, deg=1, ood=0.4\n"
        "  Combo B (attacked-optimal): sim=0.3, deg=3, ood=0.4\n\n"
        "NOTE: 0.1 and 0.2 data files use naming '0.1'/'0.2' not '0.10'/'0.20'. "
        "Combo A/B at 0.1 and 0.2 FAILED due to filename mismatch -- pending re-run."
    )

    # Combo A
    pdf.sub_title("Combo A: sim=0.6, deg=1, ood=0.4 (clean-optimal)")
    ca = combo_results["A"]["rows"]
    ca_headers = ["Perturbation", "Per-Seed Accuracies", "Mean +/- Std"]
    ca_rows = []
    for ptb, accs, mean, std in ca:
        if accs is None:
            ca_rows.append([f"{ptb}", "FAILED (file not found)", "N/A"])
        else:
            acc_str = ", ".join([f"{a:.4f}" for a in accs])
            ca_rows.append([f"{ptb}", acc_str, f"{mean:.4f} +/- {std:.4f}" if mean is not None else "N/A"])
    pdf.result_table(ca_headers, ca_rows)

    # Combo B
    pdf.sub_title("Combo B: sim=0.3, deg=3, ood=0.4 (attacked-optimal)")
    cb = combo_results["B"]["rows"]
    cb_rows = []
    for ptb, accs, mean, std in cb:
        if accs is None:
            cb_rows.append([f"{ptb}", "FAILED (file not found)", "N/A"])
        else:
            acc_str = ", ".join([f"{a:.4f}" for a in accs])
            cb_rows.append([f"{ptb}", acc_str, f"{mean:.4f} +/- {std:.4f}" if mean is not None else "N/A"])
    pdf.result_table(ca_headers, cb_rows)

    # ── GPPT Baseline Comparison ─────────────────────────────────
    pdf.section_title("7. Comparison: Best Results vs GPPT Baseline")
    pdf.body_text(
        "GPPT baseline (from Meeting 10, paper-aligned code):"
    )

    gppt_row = ["GPPT"] + [f"{GPPT[p]:.4f}" for p in ptbs]
    # Build best combo rows from available data
    compare_rows = [gppt_row]
    for combo_name, combo_info in combo_results.items():
        row = [combo_info["label"][:45]]
        for ptb in ptbs:
            match = [r for r in combo_info["rows"] if r[0] == ptb]
            if match and match[0][2] is not None:
                row.append(f"{match[0][2]:.4f}")
            else:
                row.append("---")
        compare_rows.append(row)

    compare_headers = ["Method"] + ptb_labels
    pdf.result_table(compare_headers, compare_rows)

    # Also add best single-defense results for reference
    pdf.sub_title("Best Single-Defense (for reference)")
    sd_headers = ["Defense"] + ptb_labels
    sd_rows = []
    for name, results in [("sim_pt (sim=0.6)", sim_results[4]), ("degree_pt (deg=1)", deg_results[0]), ("out_detect_pt (ood=0.4)", ood_results[1])]:
        row = [name]
        for ptb in ptbs:
            _, mean, std = results[ptb]
            row.append(f"{mean:.4f}" if mean is not None else "---")
        sd_rows.append(row)
    pdf.result_table(sd_headers, sd_rows)

    # ── Individual Seed Data ─────────────────────────────────────
    pdf.section_title("8. Individual Per-Seed Results")
    pdf.body_text("All 5 seeds shown. MyTask.py drops the lowest seed and reports mean of remaining 4.")

    for phase_name, results, ptb_subset in [
        ("Phase 3a -- sim_pt", sim_results, [0.0, 0.05]),
        ("Phase 3b -- degree_pt", deg_results, [0.0, 0.05]),
        ("Phase 3c -- out_detect_pt", ood_results, [0.0, 0.05]),
    ]:
        pdf.sub_title(phase_name)
        for ptb in ptb_subset:
            label = "Clean" if ptb == 0.0 else f"Attacked {ptb}"
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(0, 5, f"  {label}:", new_x="LMARGIN", new_y="NEXT")
            for r in results:
                accs, mean, std = r[ptb]
                if accs:
                    acc_str = "  ".join([f"seed{i+1}={a:.4f}" for i, a in enumerate(accs)])
                    pdf.set_font("Helvetica", "", 7)
                    pdf.cell(0, 4, f"    {r['param']}: {acc_str}  |  trimmed mean={mean:.4f} +/- {std:.4f}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)

    # ── Key Findings & Next Steps ─────────────────────────────────
    pdf.section_title("9. Key Findings & Next Steps")

    pdf.sub_title("Key Findings")
    findings = [
        "1. Filtering Tips thresholds are NOT dataset-universal. Paper defaults (0.4, 2, 0.5) are suboptimal for Cora across all three defense types.",
        "2. degree_pt (deg=1) is the strongest single defense on clean graphs: 0.4435 vs GPPT 0.4350. However, it underperforms on attacked graphs (0.2448).",
        "3. out_detect_pt (ood=0.4) is the most robust single defense: clean 0.4490, attacked 0.2485. Smallest clean-attacked gap (0.2005). Only defense with consistent optimal threshold across perturbation levels.",
        "4. Clean-optimal THRESHOLDS != Attacked-optimal THRESHOLDS for sim_pt and degree_pt. This means the defense strategy needs to be perturbation-aware -- a single set of thresholds cannot be optimal for both clean and attacked.",
        "5. Combined defenses (Combo A and B) underperform single defenses on clean graphs, suggesting negative interaction when defense prompts are combined without additional tuning. The interaction between multiple defense prompts needs further investigation.",
        "6. Paper-aligned code (tau_tune only, no filter_module) produces STABLE training. No NaN loss issues observed across all 40 experiments.",
    ]
    for f in findings:
        pdf.body_text(f)

    pdf.sub_title("Next Steps")
    next_steps = [
        "1. Fix filename issue: re-run Combo A and B with Meta_Self-0.1 and Meta_Self-0.2 (not 0.10/0.20).",
        "2. Tune prompt_lr and pt_threshold on top of the best Filtering Tips thresholds. Meeting 10 found prompt_lr=0.004, pt_threshold=0.25 optimal on the OLD code -- these may change on paper-aligned code.",
        "3. Investigate why combined defenses underperform single defenses. Possible causes: (a) too many nodes get defense prompts simultaneously, saturating the feature space, (b) the averaging-based fusion (no_attention) doesn't handle multiple prompt types well.",
        "4. Test cross-perturbation generalization: train on one perturbation level, evaluate on others. This is the true robustness test.",
        "5. Consider perturbation-adaptive thresholding: use different Filtering Tips thresholds based on estimated perturbation level.",
    ]
    for s in next_steps:
        pdf.body_text(s)

    # ── Appendix ──────────────────────────────────────────────────
    pdf.section_title("10. Appendix: Reproducibility Commands")
    pdf.set_font("Courier", "", 7)

    cmds = """# Phase 3a: sim_pt (degree=-1, ood=-1.0)
for sim_t in 0.2 0.3 0.4 0.5 0.6; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \\
    --pre_train_model_path '...GraphCL...lr_0.01.pth' \\
    --task NodeTask --dataset_name Cora --preprocess_method none \\
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \\
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \\
    --filter_mode original --no_attention \\
    --pt_sim_threshold ${sim_t} --pt_degree_threshold -1 --pt_out_detect_threshold -1.0 \\
    --attack_downstream --specified --attack_method Meta_Self-0.0 \\
    > logs/RobustPrompt-T/ft_sim${sim_t}_0.0_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done

# Phase 3b: degree_pt (sim=-1.0, ood=-1.0)
for deg_t in 1 2 3 5; do
  ... --pt_sim_threshold -1.0 --pt_out_detect_threshold -1.0 --pt_degree_threshold ${deg_t} ...
done

# Phase 3c: out_detect_pt (sim=-1.0, degree=-1)
for ood_t in 0.3 0.4 0.5 0.6 0.7; do
  ... --pt_sim_threshold -1.0 --pt_degree_threshold -1 --pt_out_detect_threshold ${ood_t} ...
done

# Phase 3d: Combined (note: use 0.1 and 0.2, NOT 0.10 and 0.20!)
for ptb in 0.0 0.05 0.1 0.15 0.2 0.25; do
  ... --pt_sim_threshold 0.6 --pt_degree_threshold 1 --pt_out_detect_threshold 0.4 \\
    --attack_downstream --specified --attack_method Meta_Self-${ptb} ...
done"""

    for line in cmds.split("\n"):
        pdf.cell(0, 3.5, line, new_x="LMARGIN", new_y="NEXT")

    # Save
    out_path = "meeting11_report.pdf"
    pdf.output(out_path)
    return out_path

if __name__ == "__main__":
    path = build_pdf()
    print(f"Report saved to: {path}")
    print(f"Size: {os.path.getsize(path)} bytes")
