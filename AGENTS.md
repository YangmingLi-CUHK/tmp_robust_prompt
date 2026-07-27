# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目目标

构建**鲁棒的图提示学习**（Robust Graph Prompt Learning）：在 GNN 预训练 + Prompt Learning 范式下，使下游节点分类任务在面对图结构扰动（Metattack 边污染）时仍保持稳定且可接受的分类性能。

## github仓库同步
项目采用github仓库进行跨团队协作，详见advanced_report/git_related.md

## 协作原则

1. **精准第一** — 在每次应当Review的时候，不考虑TOKEN消耗，精准第一，沿路仔仔细细仔仔细细审查所有的每一次计算和维数变化，以避免任何可能的误判和原地打转。
2. **谋定后动，敢于质疑** — 修改前先确认改动范围，用 TodoWrite 列出步骤。不确定时先问，不要猜测。如果提示词或指令存在模糊之处，在动手前必须主动确认。如果观察到异常，主动指出。
3. **简约至上** — 简约至上，最小改动，以免造成系统性混乱。工作时和输出时要注意语言的简洁、富有信息量、精准、专业。 
4. **高可读性** — 变量名清晰、逻辑平铺直叙。在逻辑缠绕而不显而易见时恰当注释。
5. **核心依赖** — PyTorch Geometric、deeprobust等。不要引入新依赖。
6. **报告数据规范** — 所有 accuracy 必须用 **全部 5 个 seed 的简单 mean ± std**，不去除最小值。可参考Codex全局html报告skill：beautiful-html-templates，默认简洁专业风。


## 项目进展

### 阶段时间线（Meeting 14 及以前的详细记录见 `reports/项目历史时间线_至Meeting14.md`）

| 时间 | 里程碑 | 一句话结论 |
|------|--------|-----------|
| 2026-05 | 代码修复 + Meeting 10 调参 | 在旧 `filter_module` 代码上调参：attention 有害、RobustPrompt-T 未显鲁棒性 |
| 2026-06-03 | 对齐论文，移除 filter_module | 训练简化为 `add_muti_pt → GNN₁ → τ_tune → GNN₂` |
| 2026-06 | Meeting 11–12 单 Filter 隔离调参 | RobustPrompt-T 首个稳定基线（旧 BB, lr=0.01） |
| 2026-06-14 | 数据链路统一 | 统一到 `data_attack_fewshot/`，修 10 个 Bug |
| 2026-06-17 | Meeting 13 RobustPrompt-I 诊断 | **`prompt_lr=0.01`（非 τ_tune）才是稳定性瓶颈**；稳定性以峰值精度为代价 |
| 2026-06-20 | Meeting 14 新 BB + 910 实验 | 新 `stable`/`peak` backbone；out_detect 最鲁棒、degree 最一致；clean vs att 最优阈值不一致 |

### Meeting 15（2026-07）——复现审计与根因定位【当前阶段】

Meeting 15 从「继续调参」转向「**逐行审计原版代码 vs 论文**」，核心结论是**论文 Table 1 的 Cora + GraphCL 结果无法复现**，且我们此前的"修复"反而偏离了原版实际运行的设计。

**逐行 diff 确认的原版 vs 我们的偏差**（`RobustPrompt_T_original.py` 头部有完整清单）：
1. **τ_tune** — 原版基于 prompt 修改后的**特征** cosine（单次 GNN forward）；我们改成 GNN embedding cosine（两次 forward）
2. **Attention** — 原版用 `F.normalize` 覆写 attention 输出（刻意"为了稳定"）→ 假 attention；我们修成了真 attention
3. **out_detect_pt** — 原版是 `pass`（从未实现）；我们完整实现了
4. **MSE loss** — 原版基于全图边（剪枝前）；我们基于剪枝后边
5. **hid_dim** — 原版 64，我们 256（prompt 影响力被稀释 4 倍）

**五项结构性缺陷（根因，`meeting15_root_cause.html` 2026-07-06）：**
1. **IA-PT 的"few-shot"实为 595 标签** — `get_psu_labels()` 从 `idx_test` 选 7×80=560 个高置信伪标签并入训练集（35→595），存在**测试集泄漏**。论文 57.82/58.82 的鲁棒性来自泄漏而非 prompt 设计。
2. **p_plus token bank（231K 参数）在 35 标签上纯过拟合** — p_plus=True clean 46.4% vs p_plus=False 34.4%；两者都远低于 linear probe 64.0%。
3. **特征空间稠密 prompt 淹没稀疏 BoW** — Cora BoW ~19 非零，prompt token 1433 全稠密，冻结 GNN 首层即遭分布漂移。
4. **KL loss（weight=0.3）反向抵消防御** — 把带防御 prompt 的节点拉回无 prompt 节点。
5. **2708 节点只退化成 4 种 prompt 组合** — sim_pt(cos≤0.2) 覆盖 66%、degree 覆盖 39%，细粒度表示被覆写。

**已排除的变量**（数据 Jaccard 1.000、Split 对齐、Metattack 参数、Cora 大小写等均干净）。其中：
- **攻击预算 Bug 已修**：`int(ptb * (adj.sum()//2))` 含自环，真实 ptb 膨胀 **+25.7%**；改为 `(adj.sum()-N)//2` 并**重生成全部 `Meta_Self_Cora_*.pt`**（对应 git 中 data 文件的改动）。
- **GCL 悖论**：ptb≥0.20 时 **linear probe（不用图结构）反而击败所有基于图的 prompt 方法**（0.127/0.120 vs GPPT 0.107/0.068 vs best combo）。高污染下问题从"识别攻击边"变成"是否还该用图"。
- **hid_dim=64 GraphCL linear probe 0.640 > 256-dim 0.626**，但仍救不了 M-0.25（≈10%）。

**Combo Filter 实验（`meeting15_part1/part2` + `_combo_v2_safe`）：** combo（sim+deg/sim+ood/deg+ood/all3）全面拉低 clean（0.46–0.52 vs 单 filter 0.51–0.66），仅 sim+deg 在 ptb=0.25 略优（0.103）。边异常检测（part2）：**sim_pt 是最好的攻击边检测器**（AUC≈0.645、F1≈0.44@ptb0.25）；filter_module 高召回低精度；τ_tune 检测很差；degree 从不标记任何边。

### 最新代码状态（2026-07-03 → 07-08，均为未提交改动）

- **`--prompt_variant ours|original`** — 新增 `prompt/RobustPrompt_T_original.py`（忠实原版：`pass`/假 attention/特征级 τ_tune/`filter_module=None`），`task.py` 按 variant 运行时分派。用于原版 vs 我们版本的对照复现。
- **`RobustPrompt-T-IA`** — 新增 prompt 类型，在 `node_task.py` 复刻 IA-PT 伪标签扩展（transductive 版，训练时扩展 `train_mask`/`y`，评估前还原）。
- **边异常检测指标** — `node_task.py` 集成 `edge_anomaly_metrics`（`pollution_diff` 对比 clean/attacked 图得到攻击边 GT，输出 τ_tune / filter_module / 各 tip 的 TPR/TNR/F1）。
- **GraphMAE 预训练** — 新增 `pretrain/GraphMAE.py`（掩码自编码，SCE loss）并接入 `MyPretrain.py`；新参数 `--mask_rate/--drop_edge_rate/--replace_rate/--loss_fn/--alpha_l`。已产出 `pre_trained_model/Cora.GraphMAE.GCN.64hidden_dim.pth`；07-07 评估：GPPT on GraphMAE backbone clean ≈ 0.615–0.644。
- **`eval_pretrain.py`** — GraphCL checkpoint 的 linear-probe 评估脚本（sklearn LogisticRegression，从文件名解析 hid_dim）。
- **NSPGCN（`Filter2_material/NSPGCN_model.py`）** — NSP 论文的原始实现（参考）。核心逻辑已提取为 `filters/nsp_filter.py` 并接入 `RobustPrompt-T-NSP`。
- **FocusedCleaner-LP Filter** — 新增 `filters/focusedcleaner_lp_filter.py`，链接预测式边过滤。`--filter_mode focusedcleaner_lp`。
- **NSP Filter** — 新增 `filters/nsp_filter.py`，邻居相似度保持边过滤。与 `RobustPrompt-T-NSP` 配合使用。
- **`RobustPrompt-T-NSP`** — 新增第五 defense tip `nsp_pt`。现已扩展为支持全部 6 个 tip（sim/degree/ood/nsp/focusedcleaner/other）。`--prompt_type RobustPrompt-T-NSP`。
- **`run_5filter_combos.sh`** — 5-filter 全排列批处理脚本（31 组合 × 6 ptb = 186 实验），通过阈值 -1 控制每个 tip 开关。

### 当前 backbone 现状

| 目录 / 文件 | 说明 |
|------|------|
| `pre_trained_model_raw/…permE.dropN…ratio_0.2…` | `stable` backbone（256-dim, Meeting 14） |
| `pre_trained_model_raw/…permE.maskN…ratio_0.3…` | `peak` backbone（256-dim, Meeting 14，主力） |
| `pre_trained_model/Cora.GraphMAE.GCN.64hidden_dim.pth` | GraphMAE 64-dim（新预训练方向） |

### 下一步 TODO

1. **运行 5-filter 全排列实验** — `bash run_5filter_combos.sh`，31 组合 × 6 ptb，评估 Filter2 组合效果。
2. **p_plus=False 测试** — 当前所有实验 p_plus=True（229K 参数），测试关了 token bank 能否改善 clean。
3. **IA-PT + 最优 filter 组合** — 用伪标签扩展（595 标签）+ 最佳 filter 组合，测试能否超越 0.55 M-0.25。

## 架构要点

### 两阶段流程

1. **Pretrain**：GNN backbone 在无标签图上自监督训练（`GraphCL` 主线；`GraphMAE` 掩码自编码为新增替代方案）
2. **Downstream**：冻结 backbone，仅训练 prompt + 线性分类头，few-shot 设定

### Prompt 类型

- **Transductive**（全图，当前主线）：`GPPT`、`RobustPrompt-T`（`--prompt_variant ours|original`）、`RobustPrompt-T-IA`（IA-PT 伪标签扩展）、`RobustPrompt-T-NSP`（5-tip 版本，支持 nsp_pt + focusedcleaner_pt）、`RobustPrompt-GPF`、`RobustPrompt-GPFplus`
- **Inductive**（k-hop 子图 batch）：`All-in-one`、`Gprompt`、`GPF`、`GPF-plus`、`RobustPrompt-I`

### RobustPrompt-T 防御机制（对齐 GPromptShield）

**防御 prompt**（通过 `--pt_*_threshold` 控制开关，设 -1 即关闭）：`sim_pt`（邻居相似度低）、`degree_pt`（低度节点）、`out_detect_pt`（OOD 边端点）、`nsp_pt`（邻居相似度保持）、`focusedcleaner_pt`（链接预测异常检测）、`other_pt`（其余节点增强）。

**训练流程**：`add_muti_pt → GNN₁ → τ_tune cosine 剪枝 → GNN₂` → CE Loss + 可选正则（MSE/KL/Constraint）

**推理流程**：`add_muti_pt → GNN`（不做任何边剪枝，对齐论文）

## 核心文件路径

### 入口脚本
- `MyPretrain.py` — 预训练入口（GraphCL / GraphMAE）
- `MyTask.py` — 下游任务入口
- `eval_pretrain.py` — GraphCL checkpoint 的 linear-probe 评估（sklearn）

### 核心模块 `prompt_graph/`

| 子模块 | 用途 | 修改频率 |
|---|---|---|
| `tasker/task.py` | BaseTask：初始化 GNN、Prompt、Optimizer（含 `prompt_variant` 分派） | 高 |
| `tasker/node_task.py` | NodeTask：节点分类训练/评估、IA-PT 扩展、边异常检测指标输出 | 高 |
| `prompt/RobustPrompt_T.py` | RobustPrompt-T（我们的增强版，含 out_detect_pt / 真 attention / embedding τ_tune） | **最高** |
| `prompt/RobustPrompt_T_original.py` | 忠实原版 GPromptShield（`pass`/假 attention/特征级 τ_tune，`filter_module=None`） | 中 |
| `prompt/RobustPrompt_T_NSP.py` | RobustPrompt-T + NSP 第五 defense tip（`nsp_pt`：邻居相似度保持检测攻击边端点） | 中 |
| `prompt/GPPTPrompt.py` | GPPT baseline | 低 |
| `pretrain/GraphMAE.py` | GraphMAE 掩码自编码预训练 | 中 |
| `filters/filter_factory.py` | Filter 注册工厂（含 `focusedcleaner_lp` / `nsp` 两个新 filter） | 中 |
| `filters/focusedcleaner_lp_filter.py` | FocusedCleaner-LP：链接预测式边过滤器，MLP encoder+inner product 预测边存在概率 | 高 |
| `filters/nsp_filter.py` | NSP Filter：邻居相似度保持边过滤器，N=(A^order)X 邻居分布 cosine 检测异常边 | 高 |
| `filters/neighbor_similarity_filter.py` | OriginalFilter / NeighborSimilarityFilter / HybridFilter | 高 |
| `utils/get_args.py` | 全部命令行参数定义 | 中 |
| `utils/edge_anomaly_metrics.py` | 攻击边检测评估（pollution_diff / TPR/TNR/F1） | 中 |
| `evaluation/RobustPromptTranductiveEva.py` | RobustPrompt-T 评估逻辑 | 中 |

### 实验性 / 已接入
- `Filter2_material/NSPGCN_model.py` — NSP 论文原始实现（参考用）。核心逻辑已提取为 `filters/nsp_filter.py`，通过 `RobustPrompt-T-NSP` 和 `filter_mode=nsp` 接入 pipeline。

### 数据与权重目录

| 目录 | 用途 |
|------|------|
| `data_attack_fewshot/` | **主力攻击数据源**（`--specified` 模式） |
| `data_fewshot/` | few-shot 划分索引与 induced graph 缓存 |
| `data_pyg/` | 默认划分的攻击数据与干净数据 |
| `pre_trained_model_raw/` | 预训练权重（主要使用） |
| `logs/` | 实验日志，按 `{prompt_type}/` 组织 |

### 可忽略的目录
`data_attack_from_default_split/`、`zcy_edge.py`、`exp_record.ipynb`、`figure_plot/`、`generate_few_shot_attack*`

### 扩展组件定位
- 新增/修改 **GNN Backbone** → `prompt_graph/model/`，在 `task.py:initialize_gnn()` 注册
- 新增/修改 **Prompt** → `prompt_graph/prompt/`，在 `task.py:initialize_prompt()` 注册
- 新增/修改 **Filter** → `prompt_graph/filters/`，在 `filter_factory.py` 注册

## 报告与参考索引

### 实验报告（`reports/` 与 `advanced_report/`）

**当前阶段（Meeting 15，2026-07）：**
| 文件 | 内容 |
|------|------|
| `advanced_report/meeting15_root_cause.html` | **根因分析**：论文 Table 1 不可复现 + 五项结构性缺陷（IA-PT 泄漏、p_plus 过拟合、稠密 prompt、KL 抵消、prompt 退化） |
| `advanced_report/meeting15_diagnostics.html` | 数据溯源、污染钻取、混淆矩阵（C3 湮灭）、预算 Bug 修复、Split 审计、GCL 悖论 |
| `reports/meeting15_part1.html` / `_combo_v2_safe.html` | Part 1 精度总表（单 filter + combo，peak BB） |
| `reports/meeting15_part2.html` / `_combo_v2_safe.html` | Part 2 边异常检测（sim_pt 为最佳检测器） |
| `reports/generate_meeting15_report.py` | Meeting 15 报告生成脚本 |

**历史（详见归档）：**
| 文件 | 内容 |
|------|------|
| `reports/项目历史时间线_至Meeting14.md` | **Meeting 14 及以前的完整时间线归档** |
| `reports/meeting10_hyperparameter_search.md` | Meeting 10 全部 6 阶段调参数据 |
| `reports/meeting13_report_20260617.html` | Meeting 13 RobustPrompt-I 稳定性诊断 |
| `reports/meeting14_full_experiment_report.html` | Meeting 14 全量实验矩阵（910 runs） |
| `reports/meeting14_prep_*.html` / `metting14_prep_GCL.md` | Meeting 14 前置：重新 GCL / 元攻击重生成 |
| `reports/command_reference.md` | 远程服务器完整运行命令 |
| `reports/experiment_configurations.md` | 已锁定的预训练配置与推荐超参数 |
| `reports/Meeting11前_实验命令备用.md` / `hyper_exp备忘.md` | Filtering Tips 隔离调参命令 / 910 runs 矩阵设计 |

### 参考文献（`reference/`）
| 文件 | 内容 |
|------|------|
| `reference/GPromptShield_修复与审查报告.txt` | 2026-05-21 修复 + 2026-06-03 对齐论文的完整记录 |
| `reference/data.md` | 统一数据路径文档、调用链路、Bug 修复记录 |
| `reference/GPromptShield Elevating Resilience in Graph Prompt Learning Against Adversarial Attacks.pdf` | GPromptShield 论文 (ICLR 2025) |
| `reference/GCL论文2020.pdf` | GraphCL 预训练论文 |
| `reference/meta_attack_ICLR2019.pdf` | Metattack 论文 |
| `reference/20260514鲁棒图提示学习项目及其历程总结.pdf` | 项目早期历程 |
| `reference/26.5.28_meeting_10_交接_rprompt调参(2).pdf` | Meeting 10 交接文档 |
| `reference/26.6.11_meeting_12.pdf` | Meeting 12 记录 |
| `reference/Meeting8_梁同学跑了各浓度的默认结果.pdf` | Meeting 8 各浓度默认结果 |


## 远程服务器

- 路径: `/home/tony/LnL/DFS_HK5`
- 环境: `conda activate LnL2`
- GPU: NVIDIA RTX 5090
- 完整运行命令见 `reports/command_reference.md`
