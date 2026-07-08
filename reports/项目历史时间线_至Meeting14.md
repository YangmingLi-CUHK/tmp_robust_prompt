# 项目历史时间线（归档：2026-05-21 → Meeting 14）

> 本文件从 `CLAUDE.md` 拆出。收录 Meeting 14 及以前的详细时间线与当时的 TODO，作为历史存档。
> Meeting 15（2026-07 起）及之后的进展见 `CLAUDE.md` 的「项目进展」章节。

---

## 2026-05-21 — GPromptShield 代码修复

对照论文 "GPromptShield: Elevating Resilience in Graph Prompt Learning Against Adversarial Attacks" (ICLR 2025) 完成三项核心修复：
1. 实现 `out_detect_pt`（基于边 cosine similarity 识别 OOD 边）
2. 修复 Self-Attention 融合机制（readout-token 模式）
3. 实现 τ_tune 动态边剪枝（两阶段 GNN forward）

> 详见 `reference/GPromptShield_修复与审查报告.txt`

**注：** Meeting 15 逐行 diff 后确认这三项"修复"实际上偏离了原版论文代码的真实设计（原版 out_detect_pt 是 `pass`、attention 被 normalize 覆写、τ_tune 基于特征而非 embedding）。详见 CLAUDE.md 当前进展。

## 2026-05-28 — Meeting 10：大规模调参（旧代码）

在含 `filter_module` 的旧代码上进行了 6 阶段超参数搜索。核心发现：**attention 有害、p_plus 有益、正则效果有限、RobustPrompt-T 未展现鲁棒性**（best clean 0.34 vs GPPT 0.44）。

> 详见 `reports/meeting10_hyperparameter_search.md`

## 2026-05-29 — NaN Loss 修复

修复三个问题：Double Softmax（CrossEntropyLoss 冲突）、梯度裁剪（KL/MSE 导致梯度爆炸）、UnboundLocalError（NaN 时 test_acc 未赋值）。

## 2026-06-03 — 严格对齐论文，移除 filter_module

经重新审查论文原文，确认 `filter_module`（基于原始特征/AX 余弦的边过滤）为代码额外引入，论文中不存在。**移除 filter_module**，训练流程简化为：`add_muti_pt → GNN₁ → τ_tune 剪枝 → GNN₂`。论文对齐后的代码于 Meeting 11–14 期间开始跑实验。

> 详见 `reference/GPromptShield_修复与审查报告.txt` Section 六–八

## 2026-06-11 — Meeting 11–12：Filtering Tips 单 Filter 隔离调参 + RobustPrompt-T 基线

启动论文 Section 4.2 的 Filtering Tips 分选阈值调参。采用隔离策略（每次只激活一种 defense prompt，其余阈值设为不可能触发的值），在干净图和 0.05 污染图上分别扫 sim/degree/ood 阈值。Meeting 12 产出 RobustPrompt-T 首个稳定基线（旧 backbone, lr=0.01）：best clean 0.449 (ood=0.4)，best att 0.05 为 0.249 (ood=0.4)，全部 5/5 seed 稳定。

> 详见 `reference/26.6.11_meeting_12.pdf`、`reports/Meeting11前_实验命令备用.md`

## 2026-06-14 — 数据链路统一与 Bug 修复

统一预训练与下游数据源到 `data_attack_fewshot/`，修复数据加载链路的 10 个 Bug，Cora LCC 一致性验证通过。

> 详见 `reference/data.md`

## 2026-06-17→06-18 — Meeting 13：RobustPrompt-I 稳定性诊断

对 RobustPrompt-I（Inductive 版本）进行四步消融实验，定位 NaN 不稳定性根因。**关键突破：`prompt_lr=0.01`（而非 τ_tune）是稳定性瓶颈**——lr 从 0.001 提升到 0.01 后 clean 稳定 seed 从 1/5 跃至 4/5。发现 MSE 正则全面有毒，KL 仅在小值 (0.001) 下安全。但稳定性提升以峰值精度为代价（clean 0.615→0.452）。pt_threshold 是精度-鲁棒性 trade-off 的直接控制杆。

> 详见 `reports/meeting13_report_20260617.html` | 日志: `logs/RobustPrompt-I/`

## 2026-06-20 — Meeting 14：新 Backbone + 全量单 Filter 实验矩阵

**重做 GCL 预训练**，产出两个新 backbone（替代旧的单一 lr=0.01 backbone）：

| 标签 | 配置 | Linear Probe Clean |
|------|------|--------------------|
| `stable` | permE/dropN, lr=0.001, ratio=0.2, seed=1 | 0.5689 |
| `peak` | permE/maskN, lr=0.001, ratio=0.3, seed=1 | 0.6262 |

在论文对齐代码（纯 τ_tune，无 filter_module）上跑 **910 次实验**：3 种单 Filter (sim/degree/ood) × 各 4-5 个阈值 × 2 BB × 6 ptb × 5 seeds。核心发现：

- **新 GPPT baseline 大幅提升**：stable 0.601, peak 0.658（旧 backbone 仅 0.435）
- **out_detect_pt 是最鲁棒的单 filter**：clean 0.676 (stable, ood=0.5)，att 0.05 达 0.289 (peak, ood=0.5)
- **degree_pt 是唯一跨扰动一致的 defense**：deg=1 在 6/6 ptb 全部 5/5 稳定
- **sim=0.2 + stable BB 系统性不稳定**（11 例 NaN 中占 8 例），应避免
- **Clean vs Attacked 最优阈值仍不一致**：确认 Meeting 10 结论

> 详见 `reports/meeting14_full_experiment_report.html` | 日志: `logs/baselines/`, `logs/single_filter_*/`

## Meeting 14 收尾时的 TODO（已被 Meeting 15 取代）

1. **Combo 验证** — 在 peak BB 上跑最优单 filter 组合 (sim=0.3, deg=1, ood=0.5)，验证多头是否负交互
   → Meeting 15 已完成（`meeting15_part1/part2` 及 `_combo_v2_safe`）。结论：combo 全面拉低 clean（0.46–0.52 vs 单 filter 0.51–0.66），仅 sim+deg 在 ptb=0.25 略优。
2. **高污染浓度策略** — ptb≥0.15 下所有方法崩盘，需要 perturbation-adaptive pt_threshold 或专门策略
   → Meeting 15 定位为「GCL 悖论」：ptb≥0.20 时 linear probe（不用图结构）反而胜过所有基于图的 prompt 方法。
3. **RobustPrompt-I 峰值追回** — 在 lr=0.01 稳定基础上通过调节 pt_threshold/defense 阈值追回峰值
4. **I 版本的 lr=0.01 发现在 T 版本上验证**
