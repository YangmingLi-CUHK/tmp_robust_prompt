Phase 3a — 扫 sim（干净图 0.00 + 污染图 0.05）：


# 干净图 (Meta_Self-0.00)
for sim_t in 0.2 0.3 0.4 0.5 0.6; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --pt_sim_threshold ${sim_t} \
    --attack_downstream --specified --attack_method Meta_Self-0.0 \
    > logs/RobustPrompt-T/ft_sim${sim_t}_0.0_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done

# 污染图 0.05
for sim_t in 0.2 0.3 0.4 0.5 0.6; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --pt_sim_threshold ${sim_t} \
    --attack_downstream --specified --attack_method Meta_Self-0.05 \
    > logs/RobustPrompt-T/ft_sim${sim_t}_0.05_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done
Phase 3b — 固定 best sim，扫 degree（干净 + 0.05）：


BEST_SIM=0.4  # 替换为 3a 最优值

for deg_t in 1 2 3 5; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --pt_sim_threshold ${BEST_SIM} --pt_degree_threshold ${deg_t} \
    --attack_downstream --specified --attack_method Meta_Self-0.00 \
    > logs/RobustPrompt-T/ft_deg${deg_t}_0.00_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done

for deg_t in 1 2 3 5; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    ... --pt_sim_threshold ${BEST_SIM} --pt_degree_threshold ${deg_t} \
    --attack_downstream --specified --attack_method Meta_Self-0.05 \
    > logs/RobustPrompt-T/ft_deg${deg_t}_0.05_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done
Phase 3c — 固定 best sim + best deg，扫 ood（干净 + 0.05）：


BEST_SIM=0.4   # 替换为 3a 最优值
BEST_DEG=2     # 替换为 3b 最优值

for ood_t in 0.3 0.4 0.5 0.6 0.7; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --pt_sim_threshold ${BEST_SIM} --pt_degree_threshold ${BEST_DEG} \
    --pt_out_detect_threshold ${ood_t} \
    --attack_downstream --specified --attack_method Meta_Self-0.00 \
    > logs/RobustPrompt-T/ft_ood${ood_t}_0.00_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done

for ood_t in 0.3 0.4 0.5 0.6 0.7; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    ... --pt_sim_threshold ${BEST_SIM} --pt_degree_threshold ${BEST_DEG} \
    --pt_out_detect_threshold ${ood_t} \
    --attack_downstream --specified --attack_method Meta_Self-0.05 \
    > logs/RobustPrompt-T/ft_ood${ood_t}_0.05_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done










三个阈值都在 RobustPrompt_T.py 的 add_muti_pt 里，对应论文 Section 4.2 的三个 Filtering Tips，控制哪些节点被分配哪种 defense prompt：

pt_sim_threshold (当前 0.4) — 邻居相似度阈值

# RobustPrompt_T.py:175-185
csim = 每个节点所有邻居边的平均 cosine similarity
node_use_sim_pt = csim <= 0.4 的节点  # 邻居之间特征不一致 → 可能是攻击边影响 → 给 sim_pt
直觉：正常图中邻居特征应该相似（同质性假设）。如果一个节点和邻居的平均相似度很低，说明它可能被对抗边包围了，给它 sim_pt 防御 prompt。

阈值调高（如 0.6）→ 更多节点被判为"可疑" → 更多节点获得 sim_pt
阈值调低（如 0.2）→ 只有极不相似的节点才获得 sim_pt
pt_degree_threshold (当前 2) — 度数阈值

# RobustPrompt_T.py:207-210
deg = 每个节点的度数
node_use_degree_pt = deg <= 2 的节点  # 低度节点 → 给 degree_pt
直觉：低度节点在图中信息少，更容易被攻击边影响其表示。给它们 degree_pt 增强表示能力。

阈值调高（如 5）→ 更多节点被判为"低度" → 更多节点获得 degree_pt
阈值调低（如 1）→ 只有度=1 的孤立边缘节点获得 degree_pt
pt_out_detect_threshold (当前 0.5) — OOD 边检测阈值

# RobustPrompt_T.py:231-237
每条边的 cosine similarity = x[u] · x[v]
ood_edge = cosine <= 0.5 的边  # 两端节点特征极不相似 → 可疑对抗边
node_use_ood_pt = 所有 OOD 边的端点（去重）
直觉：对抗攻击（如 Metattack）会添加连接特征不相似节点的边。这些边的 cosine 很低。OOD 边两端的节点获得 out_detect_pt。

阈值调高（如 0.7）→ 更多边被判为 OOD → 更多节点获得 out_detect_pt
阈值调低（如 0.3）→ 只有非常可疑的边才判 OOD
总结
参数	筛选条件	给什么 prompt	调高后果
sim_threshold	邻居平均相似度 ≤ 它	sim_pt	更多节点获得 sim_pt
degree_threshold	度数 ≤ 它	degree_pt	更多节点获得 degree_pt
ood_threshold	边 cosine ≤ 它	out_detect_pt	更多节点获得 out_detect_pt
三个阈值的默认值直接来自论文，在此之前从未被调过。这就是你说的 "filter 参数没有调" 的核心内容。








对，应该逐个隔离调参。把另外两个阈值设到不可能触发的值，这样每个 phase 只测一种 defense prompt 的独立效果。

重写三组命令：


# ============================================================
# Phase 3a: 单独调 sim_pt（静默 degree 和 ood）
# degree=-1 → 没有节点度数≤-1 → degree_pt 不触发
# ood=-1.0  → 没有边 cosine≤-1.0 → out_detect_pt 不触发
# ============================================================

for sim_t in 0.2 0.3 0.4 0.5 0.6; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --pt_sim_threshold ${sim_t} \
    --pt_degree_threshold -1 --pt_out_detect_threshold -1.0 \
    --attack_downstream --specified --attack_method Meta_Self-0.0 \
    > logs/RobustPrompt-T/ft_sim${sim_t}_0.0_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done

for sim_t in 0.2 0.3 0.4 0.5 0.6; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --pt_sim_threshold ${sim_t} \
    --pt_degree_threshold -1 --pt_out_detect_threshold -1.0 \
    --attack_downstream --specified --attack_method Meta_Self-0.05 \
    > logs/RobustPrompt-T/ft_sim${sim_t}_0.05_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done


# ============================================================
# Phase 3b: 单独调 degree_pt（静默 sim 和 ood）
# sim=-1.0  → 没有节点 csim≤-1.0 → sim_pt 不触发
# ood=-1.0  → 没有边 cosine≤-1.0 → out_detect_pt 不触发
# ============================================================

for deg_t in 1 2 3 5; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --pt_sim_threshold -1.0 --pt_out_detect_threshold -1.0 \
    --pt_degree_threshold ${deg_t} \
    --attack_downstream --specified --attack_method Meta_Self-0.0 \
    > logs/RobustPrompt-T/ft_deg${deg_t}_0.0_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done

for deg_t in 1 2 3 5; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --pt_sim_threshold -1.0 --pt_out_detect_threshold -1.0 \
    --pt_degree_threshold ${deg_t} \
    --attack_downstream --specified --attack_method Meta_Self-0.05 \
    > logs/RobustPrompt-T/ft_deg${deg_t}_0.05_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done


# ============================================================
# Phase 3c: 单独调 out_detect_pt（静默 sim 和 degree）
# sim=-1.0  → sim_pt 不触发
# degree=-1 → degree_pt 不触发
# ============================================================

for ood_t in 0.3 0.4 0.5 0.6 0.7; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --pt_sim_threshold -1.0 --pt_degree_threshold -1 \
    --pt_out_detect_threshold ${ood_t} \
    --attack_downstream --specified --attack_method Meta_Self-0.0 \
    > logs/RobustPrompt-T/ft_ood${ood_t}_0.0_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done

for ood_t in 0.3 0.4 0.5 0.6 0.7; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --pt_sim_threshold -1.0 --pt_degree_threshold -1 \
    --pt_out_detect_threshold ${ood_t} \
    --attack_downstream --specified --attack_method Meta_Self-0.05 \
    > logs/RobustPrompt-T/ft_ood${ood_t}_0.05_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done
每组实验的结果查看：


# 3a 结果
for f in logs/RobustPrompt-T/ft_sim*_0.0_*.log; do
  sim=$(echo "$f" | grep -oP 'ft_sim\K[0-9.]+')
  acc=$(grep "Final True Accuracy" "$f" | tail -1 | grep -oP '[\d.]+')
  echo "sim=$sim  acc_0.0=$acc"
done

# 3b 结果
for f in logs/RobustPrompt-T/ft_deg*_0.0_*.log; do
  deg=$(echo "$f" | grep -oP 'ft_deg\K[0-9]+')
  acc=$(grep "Final True Accuracy" "$f" | tail -1 | grep -oP '[\d.]+')
  echo "deg=$deg  acc_0.0=$acc"
done

# 3c 结果
for f in logs/RobustPrompt-T/ft_ood*_0.0_*.log; do
  ood=$(echo "$f" | grep -oP 'ft_ood\K[0-9.]+')
  acc=$(grep "Final True Accuracy" "$f" | tail -1 | grep -oP '[\d.]+')
  echo "ood=$ood  acc_0.0=$acc"
done

# 3a 结果
for f in logs/RobustPrompt-T/ft_sim*_0.05_*.log; do
  sim=$(echo "$f" | grep -oP 'ft_sim\K[0-9.]+')
  acc=$(grep "Final True Accuracy" "$f" | tail -1 | grep -oP '[\d.]+')
  echo "sim=$sim  acc_0.0=$acc"
done

# 3b 结果
for f in logs/RobustPrompt-T/ft_deg*_0.05_*.log; do
  deg=$(echo "$f" | grep -oP 'ft_deg\K[0-9]+')
  acc=$(grep "Final True Accuracy" "$f" | tail -1 | grep -oP '[\d.]+')
  echo "deg=$deg  acc_0.0=$acc"
done

# 3c 结果
for f in logs/RobustPrompt-T/ft_ood*_0.05_*.log; do
  ood=$(echo "$f" | grep -oP 'ft_ood\K[0-9.]+')
  acc=$(grep "Final True Accuracy" "$f" | tail -1 | grep -oP '[\d.]+')
  echo "ood=$ood  acc_0.0=$acc"
done







# Combo A: clean-optimal (sim=0.6, deg=1, ood=0.4)
for ptb in 0.0 0.05 0.10 0.15 0.20 0.25; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --pt_sim_threshold 0.6 --pt_degree_threshold 1 --pt_out_detect_threshold 0.4 \
    --attack_downstream --specified --attack_method Meta_Self-${ptb} \
    > logs/RobustPrompt-T/ft_comboA_${ptb}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done

# Combo B: attacked-optimal (sim=0.3, deg=3, ood=0.4)
for ptb in 0.0 0.05 0.10 0.15 0.20 0.25; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --pt_sim_threshold 0.3 --pt_degree_threshold 3 --pt_out_detect_threshold 0.4 \
    --attack_downstream --specified --attack_method Meta_Self-${ptb} \
    > logs/RobustPrompt-T/ft_comboB_${ptb}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done
