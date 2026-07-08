# 常用命令参考

> 最后更新: 2026-06-03
> 远程服务器: `/home/tony/LnL/DFS_HK2` | 环境: `conda activate LnL2` | GPU: NVIDIA RTX 5090

---

## 一、2026-06-03 忠于论文实验（纯 τ_tune，无 filter_module）

代码已对齐论文 GPromptShield：
- Training: `add_muti_pt → GNN₁ → τ_tune (cosine 剪枝) → GNN₂`
- Eval: `add_muti_pt → GNN`（不剪枝）
- filter_module 已注释，不参与任何边剪枝

### Round 1: 论文默认参数跑全浓度 baseline

```bash
# 论文默认参数（p_plus=True, use_attention=True, cosine_constraint=True）
# prompt_lr 使用默认 0.01（与 GPF/GPF-plus 一致）
for ptb in 0.00 0.05 0.10 0.15 0.20 0.25; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original \
    --attack_downstream --specified --attack_method Meta_Self-${ptb} \
    > logs/RobustPrompt-T/paper_attacked_${ptb}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done
```

### Round 2: no_attention + 论文默认参数

```bash
for ptb in 0.00 0.05 0.10 0.15 0.20 0.25; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --attack_downstream --specified --attack_method Meta_Self-${ptb} \
    > logs/RobustPrompt-T/paper_noatt_${ptb}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done
```

### Round 3: Filtering Tips 分选阈值调参

论文 GPromptShield 的三个 Filtering Tips 控制哪些节点获得哪种 defense prompt：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--pt_sim_threshold` | 0.4 | 邻居平均 cosine 阈值，≤此值 → sim_pt |
| `--pt_degree_threshold` | 2 | 度数阈值，≤此值 → degree_pt |
| `--pt_out_detect_threshold` | 0.5 | 边 cosine 阈值，≤此值 → out_detect_pt |

全组合 = 5×4×5 = 100 组，采用分阶段贪心策略（~20 组）：

**Phase 3a: 固定 deg=2, ood=0.5，扫 sim（论文其余默认，先跑 clean）**

```bash
for sim_t in 0.2 0.3 0.4 0.5 0.6; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --pt_sim_threshold ${sim_t} \
    > logs/RobustPrompt-T/ft_sim${sim_t}_clean_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done
```

**Phase 3b: 固定 best sim + ood=0.5，扫 degree**

```bash
for deg_t in 1 2 3 5; do
  ... --pt_sim_threshold ${BEST_SIM} --pt_degree_threshold ${deg_t} ...
done
```

**Phase 3c: 固定 best sim + best deg，扫 ood**

```bash
for ood_t in 0.3 0.4 0.5 0.6 0.7; do
  ... --pt_sim_threshold ${BEST_SIM} --pt_degree_threshold ${BEST_DEG} \
      --pt_out_detect_threshold ${ood_t} ...
done
```

**Phase 3d: 取最优组合，在 0.05 和 0.10 污染图上验证**

```bash
for ptb in 0.05 0.10; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    ... --pt_sim_threshold ${BEST_SIM} --pt_degree_threshold ${BEST_DEG} \
        --pt_out_detect_threshold ${BEST_OOD} \
    --attack_downstream --specified --attack_method Meta_Self-${ptb} \
    > logs/RobustPrompt-T/ft_best_${ptb}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done
```

### Round 4: 最优参数全浓度验证 0.00–0.25

注意：如果 Round 3 结果不理想，可能需要先回到 prompt_lr × pt_threshold 的网格搜索，在论文对齐代码上重新找到好的训练参数基线，然后再调 Filtering Tips。

---

## 二、预训练（生成新的 GraphCL 权重）

```bash
python MyPretrain.py --task GraphCL --dataset_name Cora --gnn_type GCN \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 56 --device 0 \
    --aug1 dropN --aug2 permE --lr 0.01
```

---

## 三、关键参数说明

| 参数 | 说明 |
|------|------|
| `--specified` | **必须加**，否则加载默认划分的攻击数据而非指定 shot/split 的数据 |
| `--attack_method` | 格式 `{攻击方式}-{污染浓度}`，如 `Meta_Self-0.05` |
| `--filter_mode` | original / neighbor_similarity / hybrid（2026-06-03 起 filter_module 不参与 RobustPrompt-T 边剪枝，仅保留参数兼容） |
| `--no_attention` | 关闭 Self-Attention 融合。Meeting 10 发现 attention 有害，建议开启此 flag |
