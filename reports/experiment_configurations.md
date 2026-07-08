# 已锁定的配置与推荐超参数

> 最后更新: 2026-06-03
> 关联报告: `report/meeting10_hyperparameter_search.md`
> ⚠️ Meeting 10 推荐超参数来自旧代码（含 filter_module），在新代码（纯 τ_tune）上需重新验证。

---

## 一、已锁定的预训练配置

经过超参数网格搜索，**标准 GraphCL 预训练配置**已锁定：

```
aug1=dropN  aug2=permE  ratio=0.3  lr=0.01  epochs=200
```

主权重文件：`pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth`

两个权重目录 `pre_trained_model/` 和 `pre_trained_model_raw/` 均已包含完整的 18 个 GraphCL 权重文件，覆盖全部网格组合：
- aug1 ∈ {dropN, permE, maskN} × aug2 ∈ {dropN, permE, maskN} × lr ∈ {0.005, 0.01}
- 均使用 GCN backbone，256 hidden dim，epochs=200
- 另各有 1 个 64 dim 的辅助权重（GraphCL 与 GraphMAE）

---

## 二、数据准备情况

`data_attack_fewshot/Cora/shot_5/1/Meta_Self/raw/` 下已具备全部 6 个污染浓度的攻击数据：

| ptb | 文件 | 大小 |
|---|---|---|
| 0.00 | Meta_Self_Cora_0.0.pt | 267KB |
| 0.05 | Meta_Self_Cora_0.05.pt | 278KB |
| 0.10 | Meta_Self_Cora_0.1.pt | 288KB |
| 0.15 | Meta_Self_Cora_0.15.pt | 298KB |
| 0.20 | Meta_Self_Cora_0.2.pt | 310KB |
| 0.25 | Meta_Self_Cora_0.25.pt | 323KB |

每个浓度对应的 `_idx_train.npy`、`_idx_val.npy`、`_idx_test.npy` 也齐全。

详见 `reference/data.md` 获取完整数据链路文档。

---

## 三、推荐超参数配置（来自 Meeting 10 — 旧代码）

> ⚠️ 以下配置均在含 filter_module 的旧代码上得出。2026-06-03 移除 filter_module 对齐论文后，新代码尚未跑实验，这些参数仅作参考。

### Clean 最优（高方差）

```
prompt_lr=0.008, pt_threshold=0.30, weight_mse=0.1, weight_kl=0.1
no_attention, p_plus=True, filter_mode=original
→ 0.3427±0.0589
```

### Clean 稳定（推荐主实验）

```
prompt_lr=0.004, pt_threshold=0.25, weight_mse=0.1, weight_kl=0.1
no_attention, p_plus=True, filter_mode=original
→ 0.3213±0.0274
```

### 污染图 0.05 最优

```
prompt_lr=0.004, pt_threshold=0.0, weight_mse=0, weight_kl=0
no_attention, p_plus=True
→ 0.2275±0.0259
```

### 污染图 0.1 最优

```
prompt_lr=0.008, pt_threshold=0.3, weight_mse=0, weight_kl=0
no_attention, p_plus=True
→ 0.1832±0.0471
```

---

## 四、Filtering Tips 参数（论文 Section 4.2，待调）

当前使用论文默认值，尚未针对 Cora 调优：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `pt_sim_threshold` | 0.4 | 邻居平均 cosine 阈值，≤此值 → sim_pt |
| `pt_degree_threshold` | 2 | 度数阈值，≤此值 → degree_pt |
| `pt_out_detect_threshold` | 0.5 | 边 cosine 阈值，≤此值 → out_detect_pt |

**注意区分**：
- "Filtering Tips 阈值" = 节点分选参数，控制哪些节点获得哪种 defense prompt
- "filter_module" = 边剪枝模块（2026-06-03 已移除），基于特征 cosine 过滤边
- 两者是完全不同的概念。

调参命令见 `report/command_reference.md` Round 3。
