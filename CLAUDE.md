# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目目标

构建**鲁棒的图提示学习**（Robust Graph Prompt Learning）：在 GNN 预训练 + Prompt Learning 范式下，使下游节点分类任务在面对图结构扰动（Metattack 边污染）时仍保持稳定且可接受的分类性能。

## 实验时间线

### 2026-05-21 — GPromptShield 代码对齐修复

参考 `reference/GPromptShield_修复与审查报告.txt`，三项核心修复：
1. **实现 `out_detect_pt`** — 基于边两端节点 cosine similarity 识别 OOD 边
2. **修复 Attention 融合** — readout-token 模式，key_padding_mask 标记空 slot
3. **实现 τ_tune** — 两阶段 GNN forward：GNN₁ → cosine 剪枝 → GNN₂

### 2026-05-28 — Meeting 10：大规模调参（旧代码，已废弃）

> ⚠️ 所有 Meeting 10 实验在 **2026-06-03 对齐论文前的旧代码** 上运行。
> 旧流程为 `add_muti_pt → filter_module 剪枝 → GNN₁ → τ_tune 剪枝 → GNN₂`（两阶段边剪枝），
> 与论文 `add_muti_pt → GNN₁ → τ_tune 剪枝 → GNN₂`（单阶段）不一致。
> **数据不可作为论文对齐代码的参考。**

**关键结论（仍适用）**：
1. **attention 有害** — 开启后 Clean 0.24→0.14
2. **p_plus 有益** — 20-token bank + learned combination 优于单 prompt
3. **高 lr 偏好高 pt_threshold** — lr=0.008 时 pt=0.3 最好
4. **正则效果有限** — 仅在 clean 上有微弱提升，污染图上无帮助
5. **Clean 最优参数 ≠ 污染图最优参数**
6. Best clean 0.34 仍低于 GPPT 0.44；0.05 污染下断崖式下跌至 0.19

**GPPT Baseline（各污染浓度，论文对齐代码）**：

| 污染浓度 | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 |
|---|---|---|---|---|---|---|
| Accuracy | 0.4350 | 0.2790 | 0.0700 | 0.0740 | 0.0280 | 0.0350 |

### 2026-05-29 — NaN Loss 修复

1. **Double Softmax** — Answering head `Softmax(dim=1)` + `CrossEntropyLoss` 内部 `log_softmax` 冲突 → 改为纯 `Linear`
2. **梯度裁剪** — KL/MSE 正则导致梯度爆炸 → `clip_grad_norm_(max_norm=1.0)`
3. **UnboundLocalError** — loss 为 NaN 时 `test_acc` 未赋值 → 初始化 `test_acc = float('nan')`

### 2026-06-03 — 移除 filter_module，严格对齐论文

Training: `add_muti_pt → GNN₁ → τ_tune (cosine 剪枝) → GNN₂`
Eval: `add_muti_pt → GNN`（不剪枝）
filter_module 已注释，不参与边剪枝。论文 Filtering Tips 全部用于节点分选 prompt。

### 2026-06-04 — Meeting 11：Filtering Tips 隔离调参

**方法**：调某个阈值时其余两个置为不可能触发的值（degree=-1 或 sim/ood=-1.0），确保只测一种 defense prompt 的独立效果。固定配置：`no_attention, p_plus=True, prompt_lr=0.01, pt_threshold=0.5, weight_mse=0.1, weight_kl=0.3`。5 seeds，trimmed mean（去最低 seed）。

**Phase 3a — sim_pt 独立**（degree=-1, ood=-1.0）：

| sim | Clean (0.0) | Attacked (0.05) |
|-----|-------------|-----------------|
| **0.2** | **0.4398 ± 0.0155** | 0.2130 ± 0.0350 |
| 0.3 | 0.3580 ± 0.0294 | 0.2125 ± 0.0471 |
| 0.4 (论文) | 0.3550 ± 0.0510 | 0.1810 ± 0.0621 |
| 0.5 | 0.3548 ± 0.0888 | 0.1923 ± 0.0312 |
| 0.6 | 0.3865 ± 0.0731 | **0.2198 ± 0.0641** |

Clean 最优 sim=0.2，Attacked 最优 sim=0.6 — **方向相反**。

**Phase 3b — degree_pt 独立**（sim=-1.0, ood=-1.0）：

| deg | Clean (0.0) | Attacked (0.05) |
|-----|-------------|-----------------|
| **1** | **0.4435 ± 0.0437** | **0.2448 ± 0.0636** |
| 2 (论文) | 0.4205 ± 0.0812 | 0.2268 ± 0.0545 |
| 3 | 0.3807 ± 0.0725 | 0.2392 ± 0.0172 |
| 5 | 0.3357 ± 0.0565 | 0.2195 ± 0.0414 |

**deg=1 在 clean 和 attacked 上同时最优**，Clean 0.4435 超越 GPPT 0.4350。

**Phase 3c — out_detect_pt 独立**（sim=-1.0, degree=-1）：

| ood | Clean (0.0) | Attacked (0.05) |
|-----|-------------|-----------------|
| 0.3 | 0.3612 ± 0.0233 | 0.2375 ± 0.0365 |
| 0.4 | 0.4490 ± 0.0269 | **0.2485 ± 0.0691** |
| 0.5 (论文) | 0.4515 ± 0.0427 | 0.2428 ± 0.0587 |
| 0.6 | 0.4333 ± 0.0605 | 0.2417 ± 0.0854 |
| 0.7 | 0.4667 ± 0.0617 | 0.2428 ± 0.0574 |

ood=0.4 最鲁棒（最小 clean-attacked gap 0.2005）。ood=0.7 clean 0.4667 是所有单一 defense 最高纪录。

**Phase 3d — 组合验证**：

| ptb | Combo A (0.6,1,0.4) | Combo B (0.3,3,0.4) | GPPT |
|-----|---------------------|---------------------|------|
| 0.0 | 0.3975 ± 0.0845 | 0.3365 ± 0.0472 | 0.4350 |
| 0.05 | 0.2388 ± 0.0584 | 0.1985 ± 0.0292 | 0.2790 |
| 0.1 | FAILED (ptb格式) | FAILED | 0.0700 |
| 0.15 | 0.0872 ± 0.0318 | 0.1363 ± 0.0214 | 0.0740 |
| 0.2 | FAILED (ptb格式) | FAILED | 0.0280 |
| 0.25 | 0.0880 ± 0.0297 | 0.1653 ± 0.0079 | 0.0350 |

**组合不如单一 defense**（Combo A 0.3975 < deg=1 单独 0.4435），多种 defense prompt 同时激活存在负交互。

**Meeting 11 核心结论**：
1. 论文默认值 (0.4, 2, 0.5) 在所有维度上都非最优 — Filtering Tips 调参是必要的
2. deg=1 唯一跨扰动一致；ood=0.4 最鲁棒
3. 三种 defense 各自独立都超越或逼近 GPPT clean 0.435
4. Clean-optimal 阈值 ≠ Attacked-optimal 阈值（sim 和 ood 方向相反）

### 2026-06-11 — Meeting 12：多头负交互分析 + Inductive MD-PT 升级

**为何多头组合不如单头？** 使用 `no_attention`（平均融合）时，一个节点可能同时满足多个 defense 条件，最终 prompt 为算术平均 → 互相抵消 + 正常节点过平滑。论文自注意力本意解决此问题，但 Meeting 10 证明当前实现中 attention 有害。

**Filter 中 similarity 的逻辑区分**：

| 模块 | similarity 来源 | 判断对象 | 结果 |
|------|----------------|---------|------|
| `sim_pt` | 原始特征 x | 节点平均邻居相似度 | 加 sim_pt，**不删边** |
| `out_detect_pt` | 原始特征 x | 每条边 | 给 OOD 边两端节点加 prompt，**不删边** |
| `τ_tune` | 第一次 GNN 后 node embedding | 每条边 | **真正删边** |
| `degree_pt` | 不用 similarity | 节点度数 | 加 degree_pt，**不删边** |

**RobustPrompt-I 升级（2026-06-11 完成）**：
1. 从 `{'other_pt': 'all'}` → 四种 defense prompt（sim/degree/out_detect/other）
2. 超参数全部 CLI 可配（不再硬编码）
3. Eval 改为 add-only（推理不剪枝）
4. Tune() 实现 τ_tune 两阶段 GNN

**Transductive vs Inductive gap 分析**：MetaAttack 最擅长破坏 transductive message passing。论文在 inductive（k-hop 子图）设定下运行，子图化可稀释攻击影响——这可能是我们 transductive 结果与论文差距大的原因。

### 2026-06-16 — 代码整理

- **攻击数据加载精简**：`load4data.py` 从 ~727 行缩减到 481 行。删除了 `CustomTUDataset`、`graph_sample_and_save`、`load_data4pretrain`、`load4graph`、`load4link` 等非 Cora 链路函数。新增 `load4cora_pretrain`、`load4cora_downstream_clean` 两个专用函数。保留 `load4node_attack_shot_index`、`load4node_attack_specified_shot_index`（攻击数据加载核心）。
- **GraphTask / LinkTask 已移除**：`MyTask.py` 中对应代码路径已删除，当前仅支持 `NodeTask`。如需恢复请从 git history 找回。
- **task.py ↔ task2.py 对齐**：两个文件的 RobustPrompt-I 配置统一为参数化版本（四种 defense prompt + `self.prompt_lr`）。
- **默认超参数更新**：`get_args.py` / `task.py` / `task2.py` 默认值同步为 Meeting 11 最优值（见下方当前默认配置）。

### 2026-06-17~18 — Meeting 13：RobustPrompt-I 稳定性诊断与突破

> 📋 **完整报告（含全部表格、per-seed 数据、分析与下一步）：** [`reports/meeting13_report_20260617.html`](reports/meeting13_report_20260617.html)

**实验路线**：Phase 1 四步消融 → P1 τ_tune 排除 → P2 KL 扫描 → P3 lr 突破 → A 全浓度验证 → B lr 上探 → C pt_threshold 追峰值

**关键发现（摘要）**：
1. **🎉 prompt_lr 是稳定性关键瓶颈** — lr=0.001 下 20+ 实验均 ≤1/5 稳定；lr=0.01 首次实现 Clean 4/5 + Attacked 4/5（实验 A）。~~τ_tune~~ 非根因
2. **MSE 正则全面有毒** — 唯一 clean+attacked 均全 NaN，彻底抛弃
3. **KL 小值有效、大值有毒** — kl=0.001 是安全窗口；kl=0.1 峰值 0.633（全部实验最高）但仅 1/5
4. **高攻击浓度下稳定性衰减** — ptb≤0.1 稳定 (≥3/5)，ptb≥0.15 退化严重 (0-2/5)
5. **pt_threshold 是精度-鲁棒性 trade-off** — 低 pt (0.1) → Clean 0.476 but Att 0.202；当前平衡点 pt=0.25

**核心结论**：lr=0.01 + pt=0.25 + kl=0.001 + no MSE 是当前最优稳定配置。I 峰值 0.633 远超 T 0.449 但稳定性不足——高 lr 以峰值换稳定。下一步方向：perturbation-adaptive pt_threshold 解决 clean-attacked 阈值冲突。

**日志位置**：`logs/RobustPrompt-I/stab*_0618_*.log`, `p[123]_*_0618_*.log`, `lr001_full_*_0618_*.log`, `pt*_lr001_*_0618_*.log`

## 当前默认配置

> 以下默认值已写入 `get_args.py` / `task.py` / `task2.py`，不传参数即可使用。

```
prompt_lr=0.01, pt_threshold=0.5
weight_mse=0.1, weight_kl=0.3, weight_constraint=0.2
pt_sim_threshold=0.2, pt_degree_threshold=1, pt_out_detect_threshold=0.4
no_attention (默认), p_plus=True, cosine_constraint=True
temperature=1.0, filter_mode=original
```

| 参数 | 值 | 依据 |
|------|-----|------|
| `pt_sim_threshold` | 0.2 | sim_pt 独立 clean 最优 0.4398（vs 论文 0.4→0.3550） |
| `pt_degree_threshold` | 1 | 唯一跨扰动一致最优，clean 0.4435 + attacked 0.2448 |
| `pt_out_detect_threshold` | 0.4 | 最鲁棒（最小 clean-attacked gap），clean 0.4490 + attacked 0.2485 |
| `use_attention` | False | Meeting 10 证明有害（0.24→0.14） |

**单一 defense 最佳独立效果**（隔离调参，仅作参考）：

| Defense | 最佳阈值 | Clean (0.0) | Attacked (0.05) |
|---------|---------|-------------|-----------------|
| sim_pt only | 0.2 | 0.4398 ± 0.0155 | 0.2130 ± 0.0350 |
| degree_pt only | 1 | 0.4435 ± 0.0437 | 0.2448 ± 0.0636 |
| out_detect_pt only | 0.4 | 0.4490 ± 0.0269 | 0.2485 ± 0.0691 |
| **GPPT baseline** | — | **0.4350** | **0.2790** |

⚠️ **多头组合效果显著弱于单一 defense**（Combo A clean 0.3975 < deg=1 单独 0.4435）。在解决多头融合问题前，建议优先使用单一 defense 或仅激活一种 defense prompt。

## 架构要点

### 训练与推理流程（对齐论文 GPromptShield）

**Training（`Tune` 方法）**：
1. `add_muti_pt` — 为每个节点添加对应 defense prompt（Filtering Tips 仅用于节点分选，不做边过滤）
2. GNN₁ forward（全图，无预剪枝）→ 中间 node embedding
3. τ_tune — 基于中间 embedding 的 cosine similarity 过滤边（论文 Equation 15，唯一边剪枝）
4. GNN₂ forward（剪枝后图）→ 最终 node embedding
5. Loss = CE + weight_mse × L_s + weight_kl × L_pt + weight_constraint × L_constraint

**Eval（推理）**：`add_muti_pt` → GNN forward（不剪枝，对齐论文）。

τ_tune 属于 Indirect Amplification：训练时剪枝迫使 prompt 在不依赖可疑边的前提下学习鲁棒表示；推理时训练好的 prompt 直接利用全图结构信息。

### 四类 Defense Prompt

| Prompt | 触发条件 | 默认阈值 | 论文默认 |
|--------|---------|---------|---------|
| `sim_pt` | 邻居平均 cosine 相似度 ≤ 阈值 | 0.2 | 0.4 |
| `degree_pt` | 节点度数 ≤ 阈值 | 1 | 2 |
| `out_detect_pt` | 边两端 cosine 相似度 ≤ 阈值 | 0.4 | 0.5 |
| `other_pt` | 其余所有节点（p_plus: 20-token bank + 学习权重） | — | — |

自注意力融合（`use_attention=True`）：readout token 前置 → MultiheadAttention → key_padding_mask 屏蔽空 slot → 取 readout token 输出。**当前默认关闭**（实验证明有害）。

### 两阶段流程

1. **Pretrain**：GNN backbone 在无标签图上自监督训练（GraphCL）
2. **Downstream**：冻结 backbone，仅训练 prompt + 线性分类头（answering head），few-shot 设定

### 预训练配置（已锁定）

```
aug1=dropN  aug2=permE  ratio=0.3  lr=0.01  epochs=200
```

主权重：`pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth`

`pre_trained_model/` 和 `pre_trained_model_raw/` 各含 18 个 GraphCL 权重（aug1×aug2×lr 全组合），均 GCN backbone，256 hidden dim。

### 数据加载（2026-06-16 精简后）

**当前仅支持 Cora + NodeTask**。GraphTask/LinkTask 代码已移除（可从 git history 恢复）。

三个加载函数：

| 函数 | 用途 | 数据源 |
|------|------|--------|
| `load4cora_pretrain` | 预训练：加载清洁 Cora 图 | `data_attack_fewshot/Cora/shot_5/1/Meta_Self/` (ptb=0.0) |
| `load4cora_downstream_clean` | 清洁下游任务：加载图 + few-shot split | 同上 (ptb=0.0) + `data_fewshot/` 索引 |
| `load4node_attack_specified_shot_index` | 攻击下游任务（`--specified`） | `data_attack_fewshot/{ds}/shot_{n}/{split}/Meta_Self/` |
| `load4node_attack_shot_index` | 攻击下游任务（非 specified / adaptive） | `data_pyg/Attack_data/` 或 `data_pyg/Attack_unit_test_data/` |

三种模式由 `--specified`、`--adaptive`、`--attack_downstream` 共同决定。**当前主力：`--specified` + Cora + shot_5**（唯一覆盖全部 6 个 ptb 浓度的路径）。

**攻击文件命名**：使用 `0.0`, `0.05`, `0.1`, `0.15`, `0.2`, `0.25`（不带尾部零）。Python float `0.1` 和字符串 `0.10` 拼出不同文件名，传入 `--attack_method Meta_Self-0.1` 而非 `Meta_Self-0.10`。

**已有数据**：`data_attack_fewshot/Cora/shot_5/1/Meta_Self/` 下全部 6 个浓度 raw + processed 齐全。`data_attack_fewshot/chameleon/` 仅有 ptb=0.05 raw 文件（无 processed）。`data_attack_fewshot/Cora/shot_1/` 无 Meta_Self 目录。

### 核心文件路径

| 文件 | 用途 | 频率 |
|------|------|------|
| `MyPretrain.py` | 预训练入口 | 低 |
| `MyTask.py` | 下游任务入口（仅 NodeTask） | 高 |
| `prompt_graph/tasker/task.py` | BaseTask：GNN/Prompt/Optimizer 初始化 | 高 |
| `prompt_graph/tasker/task2.py` | BaseTask（参数化版），与 task.py 功能对齐 | 中 |
| `prompt_graph/tasker/node_task.py` | NodeTask：训练与评估，数据加载路由 | 高 |
| `prompt_graph/prompt/RobustPrompt_T.py` | RobustPrompt-T（GPromptShield transductive） | **最高** |
| `prompt_graph/prompt/RobustPrompt_I.py` | RobustPrompt-I（GPromptShield inductive） | 高 |
| `prompt_graph/prompt/GPPTPrompt.py` | GPPT baseline | 低 |
| `prompt_graph/filters/filter_factory.py` | Filter 注册工厂 | 中 |
| `prompt_graph/filters/neighbor_similarity_filter.py` | OriginalFilter / NeighborSimilarityFilter | 中 |
| `prompt_graph/utils/get_args.py` | 全部命令行参数定义 | 中 |
| `prompt_graph/evaluation/RobustPromptTranductiveEva.py` | RobustPrompt-T 评估 | 中 |
| `prompt_graph/evaluation/RobustPromptInductiveEva.py` | RobustPrompt-I 评估 | 中 |
| `prompt_graph/data/load4data.py` | 数据加载（精简后 481 行） | 中 |

### Prompt 类型速查

- **Transductive**（全图节点分类）：`GPPT`、`RobustPrompt-T`、`RobustPrompt-GPF`、`RobustPrompt-GPFplus`
- **Inductive**（k-hop 子图 batch）：`All-in-one`、`Gprompt`、`GPF`、`GPF-plus`、`RobustPrompt-I`

## 常用命令（远程服务器）

远程服务器：`/home/tony/LnL/DFS_HK2`，conda 环境 `LnL2`，GPU NVIDIA RTX 5090。

### RobustPrompt-T 全浓度实验（当前默认参数）

```bash
# 当前默认参数：no_attention, p_plus, sim=0.2, deg=1, ood=0.4
for ptb in 0.0 0.05 0.1 0.15 0.2 0.25; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original \
    --attack_downstream --specified --attack_method Meta_Self-${ptb} \
    > logs/RobustPrompt-T/default_${ptb}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done
```

### 预训练

```bash
python MyPretrain.py --task GraphCL --dataset_name Cora --gnn_type GCN \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 56 --device 0 \
    --aug1 dropN --aug2 permE --lr 0.01
```

## 已知问题与待办

**已解决 ✅**：
1. Train/Eval 剪枝对齐论文（2026-06-03）
2. RobustPrompt-I 对齐 RobustPrompt-T（2026-06-11）：out_detect_pt + add-only eval + τ_tune
3. task.py / task2.py 配置统一（2026-06-16）
4. 攻击数据加载精简（2026-06-16）：删除非 Cora 链路，缩减 ~250 行

**待解决 🔴**：
1. **多头 Prompt 组合负交互**（核心未解决）— 三种 defense prompt 同时激活效果弱于单一 defense。根因：平均融合导致互相抵消 + 过平滑。需改进 fusion 机制
2. **整体鲁棒性不足** — Clean 上单一 defense 可达 0.44-0.47 超越 GPPT，但 0.05 攻击下 best single defense 仅 0.24-0.25，仍低于 GPPT 0.279
3. **Transductive vs Inductive gap** — MetaAttack 擅长破坏 transductive MP，子图化可稀释攻击。建议后续跑 RobustPrompt-I inductive 实验对比
4. **正则项在污染图上反效果** — weight_mse/weight_kl 仅在 clean 上有微弱正面效果
5. **Clean ≠ Attacked 最优阈值** — sim_pt clean 最优 0.2 vs attacked 最优 0.6。需 perturbation-aware 策略
6. **Chameleon 数据不完整** — 仅有 ptb=0.05 raw 文件，无 processed 缓存

## 协作原则

1. **简约至上** — 不过度抽象。三个相似行不急着提取函数。只改任务需要的部分
2. **谋定后动** — 修改前先确认范围，用 TodoWrite 列出步骤。不确定时先问
3. **敢于质疑** — 指令模糊时主动确认。观察到异常主动指出
4. **高可读性** — 变量名清晰、逻辑平铺直叙。不写注释解释"做了什么"，只在 WHY 不显而易见时注释
5. **核心依赖** — PyTorch Geometric、deeprobust。不引入新依赖
