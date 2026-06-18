# 统一数据路径文档

> 最后修改: 2026-06-14 (implementation)
> 修改范围: 端到端数据加载链路（预训练 → 下游攻击 → 下游清洁）

## 统一数据源

**唯一起源**: `data_attack_fewshot/Cora/shot_5/1/Meta_Self/raw/`

| 文件 | 内容 | 形状 |
|------|------|------|
| `Cora_features.npz` | L1-归一化特征矩阵 | (2708, 1433) |
| `Cora_labels.npy` | 整数标签 (0–6) | (2708,) |
| `Meta_Self_Cora_0.0.pt` | 清洁邻接矩阵 (+ 自环) | (2708, 2708) |
| `Meta_Self_Cora_0.05.pt` | 5% 扰动邻接矩阵 | (2708, 2708) |
| `Meta_Self_Cora_0.1.pt` | 10% 扰动邻接矩阵 | (2708, 2708) |
| `Meta_Self_Cora_0.15.pt` | 15% 扰动邻接矩阵 | (2708, 2708) |
| `Meta_Self_Cora_0.2.pt` | 20% 扰动邻接矩阵 | (2708, 2708) |
| `Meta_Self_Cora_0.25.pt` | 25% 扰动邻接矩阵 | (2708, 2708) |
| `Meta_Self_Cora_{ptb}_idx_{train,val,test}.npy` | 每种浓度的 split 索引 | 各 (N,) |

各浓度的 PyG processed 缓存:
`data_attack_fewshot/Cora/shot_5/1/Meta_Self/{ptb}_processed/data_Cora_Meta_Self_{ptb}.pt`

## 特征归一化验证结论

通过数值比对验证（2026-06-14, numpy 直接比对 + KDTree 最近邻匹配）:

- Planetoid raw 特征 (ind.cora.allx+tx): bag-of-words counts, row sum 均值 ~9–23
- Attack_data `cora_features.npz`: **已 L1-归一化**, row sum = 1.0
- Planetoid 经 NormalizeFeatures → row sum = 1.0
- 归一化后的 Planetoid vs Attack_data: **fp 精度一致** (max diff ~1.7e-9, 仅浮点舍入误差)
- 边集比对: 5278 条无向边，完全一致，0 条差异

**结论: 预训练和下游使用同一份 Cora 数据，归一化一致。无数据分布不匹配问题。**

## 数据调用链路（修改后，已实现）

### 链路 1: 预训练 (GraphCL)

```
MyPretrain.py                                         [seed 已传入 GraphCL]
  → GraphCL(dataset_name='Cora', hid_dim=256, seed=seed)
    → GraphCL.load_graph_data()                       [GraphCL.py:23]
      → NodePretrain(dataname='Cora', num_parts=200)  [load4data.py:NodePretrain]
        → load4cora_pretrain()                        [load4data.py:load4cora_pretrain]
          → AttackDataset_specified(ptb_rate='0.0', transform=NormalizeFeatures())
            ← 统一数据源: data_attack_fewshot/Cora/shot_5/1/Meta_Self/raw/
          → ClusterData(data, num_parts=200)          → 200 个子图
    → pretrain(aug1, aug2, lr)
      → 保存权重: pre_trained_model_raw/
        {Cora.GraphCL.GCN.256_hidden_dim.aug1_X.aug2_Y.lr_Z.seed_N.pth}  [seed 新增]
```

### 链路 2: 下游 — 攻击图 (--specified 模式, 主力实验)

```
MyTask.py
  → NodeTask(attack_downstream=True, specified=True, attack_method='Meta_Self-0.05')
    → load_shot_attack_data()                         [node_task.py:70]
      → load4node_attack_specified_shot_index()       [load4data.py]
        → AttackDataset_specified(ptb_rate='0.05')
          ← 统一数据源: data_attack_fewshot/.../Meta_Self/raw/ + 0.05_processed/
        → 加载 train/val/test mask 从 index/          ← data_attack_fewshot/.../index/
    → RobustPrompt-T 或 GPPT 训练
```

### 链路 3: 下游 — 清洁图 (无攻击)

```
MyTask.py
  → NodeTask(attack_downstream=False)
    → load_data()                                     [node_task.py:139]
      → load4cora_downstream_clean(dataname, shot_num, run_split)  [load4data.py]
        → AttackDataset_specified(ptb_rate='0.0', transform=NormalizeFeatures())
          ← 统一数据源
        → 加载 split 索引                             ← data_fewshot/Cora/shot_5/1/index/
```

### 链路 4: 下游 — 攻击图 (非 specified 模式, 仅 ptb=0.05)

```
MyTask.py
  → NodeTask(attack_downstream=True, specified=False, attack_method='Meta_Self-0.05')
    → load_shot_attack_data()
      → load4node_attack_shot_index()                 [load4data.py]
        → get_dataset('Attack-Cora', 'Meta_Self', 0.05)
          → AttackDataset(ptb_rate='0.05')             ← data_pyg/Attack_data/Meta_Self/cora/raw/
```

### 链路 5: 攻击数据生成 (一次性)

```
generate_few_shot_attack.py
  → get_dataset('Attack-Cora', 'Meta_Self', 0.0)      ← data_pyg/Attack_data/Meta_Self/cora/raw/
  → Metattack 生成扰动图
  → 保存到 data_attack_fewshot/Cora/shot_5/1/Meta_Self/raw/
```

---

## Bug 修复记录（2026-06-14 已完成）

| # | 文件 | 修复内容 | 状态 |
|---|------|---------|------|
| 1 | `data_attack_fewshot/attackdata_specified.py` | `raw_file_names`: typo `lablels`→`labels`, 补 `.npz`/`.npy` 扩展名 | ✅ |
| 2 | `data_pyg/Attack_data/attackdata.py` | `raw_file_names`: 同上 typo + 扩展名修复 | ✅ |
| 3 | `prompt_graph/pretrain/GraphCL.py` | `__init__` 新增 `seed` 参数; `pretrain()` 保存路径加入 `seed_N` | ✅ |
| 4 | `MyPretrain.py` | 传入 `seed=seed` 给 GraphCL | ✅ |
| 5 | `MyTask.py` | trimmed mean: 过滤 NaN, `sorted()[1:-1]` 同时去高低值 | ✅ |
| 6 | `MyTask.py` | NaN 返回时不再污染汇总统计 | ✅ |
| 7 | `prompt_graph/data/load4data.py` | 新增 `load4cora_pretrain()`, `load4cora_downstream_clean()` 统一数据源 | ✅ |
| 8 | `prompt_graph/data/__init__.py` | 更新导出列表，标注已删除的函数 | ✅ |
| 9 | `prompt_graph/tasker/node_task.py` | 调用 `load4cora_downstream_clean` 替代旧函数 | ✅ |
| 10 | `data_pyg/data_pyg.py` | 精简为仅 Attack-/Unit- 分支 | ✅ |

---

## 删除的代码和目录

### 从 load4data.py 删除的函数
以下函数已从 `prompt_graph/data/load4data.py` 中删除。如需恢复，请从 git history (`git log -- prompt_graph/data/load4data.py`) 找回：
- `CustomTUDataset` class
- `graph_sample_and_save`
- `load_data4pretrain`
- `load4graph`
- `load4node_shot_index` → 替换为 `load4cora_downstream_clean`
- `load4node_demo1`
- `load4node_demo2` → 替换为 `load4cora_pretrain`
- `load4link_prediction_single_graph`
- `load4link_prediction_multi_graph`
- `node_degree_as_features`
- `load4link`

### 从 data_pyg/data_pyg.py 删除的分支
- Planetoid (Cora, CiteSeer, PubMed, DBLP)
- CitationFull
- Coauthor (CS, Physics)
- Amazon (Computers, Photo)
- Reddit, Flickr, Yelp, WikiCS
- PPI
- ogbn-arxiv

这些分支在需要加入新数据集时恢复。恢复方法: 从 git history 复制对应分支代码。

### 删除的数据目录
| 目录 | 删除原因 |
|------|---------|
| `data/Planetoid/` | 不再引用，数据源已统一到 data_attack_fewshot/ |
| `data_attack_fewshot/chameleon/` | 不完整（无 features, 无 labels, 无 processed） |
| `data_attack_fewshot/Cora/shot_1/` | 仅有 index 文件，无 Meta_Self 攻击数据 |
| `data_pyg/Attack_data/Meta_Self/cora_ml/` | 非 Cora 主线 |
| 所有 `__pycache__/` 和 `.DS_Store` | 构建产物 / OS 元数据 |

### 保留的数据目录
| 目录 | 用途 |
|------|------|
| `data_attack_fewshot/Cora/shot_5/1/Meta_Self/` | **主数据源**: raw + 所有 ptb processed |
| `data_attack_fewshot/Cora/shot_5/1/index/` | Specified 模式的 few-shot split 索引 |
| `data_attack_fewshot/Cora/shot_5/1/Meta_Self/raw/` | 清洁图 + 5 个扰动浓度的邻接矩阵 |
| `data_attack_from_default_split/` | 非 specified 模式的 split 索引 |
| `data_fewshot/` | 清洁下游任务的 split 索引 |
| `data_pyg/Attack_data/Meta_Self/cora/` | `generate_few_shot_attack.py` 的攻击基底 (ptb=0.05) |
| `data_pyg/Attack_unit_test_data/` | 自适应攻击模式（暂未使用） |

---

## 如何添加新数据集

1. 在 `data_attack_fewshot/{NewDS}/shot_5/1/Meta_Self/raw/` 下放置:
   - `{NewDS}_features.npz` — L1-归一化特征
   - `{NewDS}_labels.npy` — 标签
   - 清洁邻接矩阵和扰动邻接矩阵
2. 在 `load4data.py` 的 `load4cora_pretrain` 和 `load4cora_downstream_clean` 中添加数据集分支
3. 在 `get_args.py` 中确认 `--dataset_name` choices 包含新数据集名
4. 恢复 `data_pyg/data_pyg.py` 中的 Planetoid/CitationFull 分支 (如需要从互联网加载)

## 数据文件大小参考

| 文件 | 大小 |
|------|------|
| `Cora_features.npz` | 86 KB |
| `Cora_labels.npy` | 22 KB |
| `Meta_Self_Cora_0.0.pt` (清洁邻接) | 267 KB |
| `Meta_Self_Cora_0.05.pt` | 278 KB |
| `Meta_Self_Cora_0.25.pt` | 323 KB |
| 每个 `_processed/data_*.pt` | ~15.8 MB |
