Step 1: 快速冒烟测试（2分钟）

# 确认整个链路能跑通：单 seed 单超参预训练 1 epoch
cd /home/tony/LnL/DFS_HK2
python MyPretrain.py \
    --task GraphCL --dataset_name Cora --gnn_type GCN \
    --hid_dim 256 --num_layer 2 --epochs 5 --seed 1 --device 0 \
    --aug1 dropN --aug2 permE --lr 0.01

# 确认下游评估脚本能跑
python eval_pretrain.py --device 0 --top_k 3
Step 2: 正式 5-seed 预训练（挂后台）

# 先在本地改好 pretrain_5seed.sh（路径从 ./pre_trained_model_raw/ → 服务器路径保持一致）
# 服务器上：
cd /home/tony/LnL/DFS_HK2
nohup bash pretrain_5seed.sh > logs/pretrain_5seed_$(date +%Y%m%d_%H%M%S).log 2>&1 &
P.S. tail -f logs/pretrain_5seed_*.log

# 预计时间：90 组 × ~200 epochs ~ 视 GPU 而定，5090 上很快
Step 3: 选最佳预训练权重

cd /home/tony/LnL/DFS_HK2
python eval_pretrain.py --device 0 | tee logs/eval_pretrain_$(date +%Y%m%d_%H%M%S).log
输出会告诉你：

每个 checkpoint 的 test accuracy（LogisticRegression 线性探测）
Top-K 排名
按超参组聚合的 mean±std（跨 seed 平均，选出稳定最优的 aug1×aug2×lr 组合）
最佳 checkpoint 的完整路径，可以直接填到下游实验的 --pre_train_model_path
Step 4: 用最佳权重跑下游实验（示例）

BEST_MODEL="./pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.seed_3.pth"

for ptb in 0.0 0.05 0.1 0.15 0.2 0.25; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path "$BEST_MODEL" \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --attack_downstream --specified --attack_method Meta_Self-${ptb} \
    > logs/robust_prompt_t/best_backbone_${ptb}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done