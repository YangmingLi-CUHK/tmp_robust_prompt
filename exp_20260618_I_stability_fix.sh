#!/bin/bash
# =============================================================================
# 2026-06-18 实验：RobustPrompt-I 稳定性修复（Phase 1 消融后续）
# 服务器：/home/tony/LnL/DFS_HK2 | conda: LnL2 | GPU: RTX 5090
#
# 基于 Meeting 13 Phase 1 发现：
#   - τ_tune 对 clean 图有毒（CE-only clean ✓ → +τ_tune ✗ NaN）
#   - MSE 正则全面有毒（Step 3 全 NaN）
#   - 极小 KL 是唯一平衡配置（kl=0.001, clean+attacked 均成功）
#   - 所有成功配置仅 1/5 seed — 需提升稳定性
#
# 用法：
#   bash exp_20260618_I_stability_fix.sh 0        ← P1: τ_tune 禁用验证 (4 jobs)
#   bash exp_20260618_I_stability_fix.sh 1        ← P2: KL 正则扫描 (8 jobs)
#   bash exp_20260618_I_stability_fix.sh 2        ← P3: prompt_lr 微调 (8 jobs)
#   bash exp_20260618_I_stability_fix.sh results  ← 查看全部结果
#
# 静默逻辑（同前）：
#   sim_pt 静默：--pt_sim_threshold -1.0
#   degree_pt 静默：--pt_degree_threshold -1
#   out_detect_pt 静默：--pt_out_detect_threshold -1.0
#   τ_tune 禁用：--pt_threshold -1.1
#
# 日志位置：logs/RobustPrompt-I/
# =============================================================================

PHASE="${1:-0}"

MODEL="./pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth"
LOG_DIR="logs/RobustPrompt-I"
mkdir -p "$LOG_DIR"

# =============================================================================
# P1: τ_tune 禁用验证 — 确认 τ_tune 是否为 NaN 根因
#   去掉 MSE（已知有毒），只保留 CE-only 和 +tiny KL
#   对比 pt_threshold=-1.1 (禁用 τ_tune) vs pt_threshold=0.25 (启用)
# =============================================================================
run_p1_tautune_ablation() {
  echo "====== P1: τ_tune Ablation (4 jobs, sequential) ======"
  START_TIME=$(date +%s)

  # P1a: CE-only + τ_tune DISABLED (对比 Phase 1 Step 1 的失败)
  echo "--- P1a: CE-only, NO tau_tune (pt=-1.1) ---"
  for ptb in 0.0 0.05; do
    LOG="$LOG_DIR/p1_CEonly_noTau_${ptb}_$(date +%m%d_%H%M).log"
    echo "  -> ptb=$ptb  log=$(basename $LOG)"
    CUDA_VISIBLE_DEVICES=0 python MyTask.py \
      --pre_train_model_path "$MODEL" \
      --task NodeTask --dataset_name Cora --preprocess_method none \
      --gnn_type GCN --prompt_type RobustPrompt-I --shot_num 5 --run_split 1 \
      --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
      --filter_mode original \
      --attack_downstream --specified \
      --prompt_lr 0.001 --pt_threshold -1.1 \
      --weight_mse 0 --weight_kl 0 --weight_constraint 0 \
      --pt_sim_threshold 0.2 --pt_degree_threshold 1 --pt_out_detect_threshold 0.4 \
      --attack_method Meta_Self-${ptb} \
      > "$LOG" 2>&1
    grep -q "Final True Accuracy" "$LOG" && echo "  <- OK" || echo "  <- NaN"
  done

  # P1b: +tiny KL + τ_tune DISABLED (对比 Phase 1 Step 4)
  echo "--- P1b: +tiny KL, NO tau_tune (pt=-1.1) ---"
  for ptb in 0.0 0.05; do
    LOG="$LOG_DIR/p1_KL_noTau_${ptb}_$(date +%m%d_%H%M).log"
    echo "  -> ptb=$ptb  log=$(basename $LOG)"
    CUDA_VISIBLE_DEVICES=0 python MyTask.py \
      --pre_train_model_path "$MODEL" \
      --task NodeTask --dataset_name Cora --preprocess_method none \
      --gnn_type GCN --prompt_type RobustPrompt-I --shot_num 5 --run_split 1 \
      --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
      --filter_mode original \
      --attack_downstream --specified \
      --prompt_lr 0.001 --pt_threshold -1.1 \
      --weight_mse 0 --weight_kl 0.001 --weight_constraint 0 \
      --pt_sim_threshold 0.2 --pt_degree_threshold 1 --pt_out_detect_threshold 0.4 \
      --attack_method Meta_Self-${ptb} \
      > "$LOG" 2>&1
    grep -q "Final True Accuracy" "$LOG" && echo "  <- OK" || echo "  <- NaN"
  done

  ELAPSED=$(( $(date +%s) - START_TIME ))
  echo "P1 done in ${ELAPSED}s."
  echo ""
  echo "解读指南："
  echo "  如果 P1a clean 5/5 稳定 → τ_tune 确认为 clean 图 NaN 根因"
  echo "  如果 P1a attacked 仍 NaN → attacked 图需要 τ_tune 之外的额外保护"
  echo "  如果 P1b 5/5 稳定 → 最佳方案: KL 正则 + 禁用 τ_tune"
}

# =============================================================================
# P2: KL 正则扫描 — 找最优 KL 强度
#   固定: τ_tune 开启 (pt=0.25), no MSE
#   扫描: kl ∈ {0.005, 0.01, 0.05, 0.1}
# =============================================================================
run_p2_kl_scan() {
  echo "====== P2: KL Regularization Scan (8 jobs, sequential) ======"
  START_TIME=$(date +%s)

  for kl in 0.005 0.01 0.05 0.1; do
    echo "--- KL=$kl (tau_tune ON, no MSE) ---"
    for ptb in 0.0 0.05; do
      LOG="$LOG_DIR/p2_KL${kl}_${ptb}_$(date +%m%d_%H%M).log"
      echo "  -> KL=$kl ptb=$ptb  log=$(basename $LOG)"
      CUDA_VISIBLE_DEVICES=0 python MyTask.py \
        --pre_train_model_path "$MODEL" \
        --task NodeTask --dataset_name Cora --preprocess_method none \
        --gnn_type GCN --prompt_type RobustPrompt-I --shot_num 5 --run_split 1 \
        --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
        --filter_mode original \
        --attack_downstream --specified \
        --prompt_lr 0.001 --pt_threshold 0.25 \
        --weight_mse 0 --weight_kl ${kl} --weight_constraint 0 \
        --pt_sim_threshold 0.2 --pt_degree_threshold 1 --pt_out_detect_threshold 0.4 \
        --attack_method Meta_Self-${ptb} \
        > "$LOG" 2>&1
      valid=$(grep -c "Final True Accuracy" "$LOG")
      echo "  <- valid seeds with acc: $valid/5"
    done
  done

  ELAPSED=$(( $(date +%s) - START_TIME ))
  echo "P2 done in ${ELAPSED}s."
  echo ""
  echo "解读指南："
  echo "  找 ≥3/5 seed 稳定的 KL 值"
  echo "  如果 KL≥0.05 仍仅 1/5 → KL 不是稳定性瓶颈，回到 τ_tune"
}

# =============================================================================
# P3: prompt_lr 微调 — 排除 lr 过低导致的不稳定
#   固定: τ_tune 开启, KL=best from P2 (默认 0.001), no MSE
#   扫描: prompt_lr ∈ {0.005, 0.01}
# =============================================================================
run_p3_lr_scan() {
  echo "====== P3: prompt_lr Scan (8 jobs, sequential) ======"
  START_TIME=$(date +%s)

  for lr in 0.005 0.01; do
    echo "--- prompt_lr=$lr (tau_tune ON, KL=0.001, no MSE) ---"
    for ptb in 0.0 0.05; do
      LOG="$LOG_DIR/p3_lr${lr}_${ptb}_$(date +%m%d_%H%M).log"
      echo "  -> lr=$lr ptb=$ptb  log=$(basename $LOG)"
      CUDA_VISIBLE_DEVICES=0 python MyTask.py \
        --pre_train_model_path "$MODEL" \
        --task NodeTask --dataset_name Cora --preprocess_method none \
        --gnn_type GCN --prompt_type RobustPrompt-I --shot_num 5 --run_split 1 \
        --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
        --filter_mode original \
        --attack_downstream --specified \
        --prompt_lr ${lr} --pt_threshold 0.25 \
        --weight_mse 0 --weight_kl 0.001 --weight_constraint 0 \
        --pt_sim_threshold 0.2 --pt_degree_threshold 1 --pt_out_detect_threshold 0.4 \
        --attack_method Meta_Self-${ptb} \
        > "$LOG" 2>&1
      valid=$(grep -c "Final True Accuracy" "$LOG")
      echo "  <- valid seeds with acc: $valid/5"
    done
  done

  ELAPSED=$(( $(date +%s) - START_TIME ))
  echo "P3 done in ${ELAPSED}s."
}

# =============================================================================
# 结果汇总
# =============================================================================
show_results() {
  echo "====== All Results Summary ======"
  echo ""

  for prefix in p1_CEonly_noTau p1_KL_noTau p2_KL0.005 p2_KL0.01 p2_KL0.05 p2_KL0.1 p3_lr0.005 p3_lr0.01; do
    echo "--- $prefix ---"
    for ptb in 0.0 0.05; do
      latest=$(ls -t "$LOG_DIR/${prefix}_${ptb}_"*.log 2>/dev/null | head -1)
      if [ -z "$latest" ]; then
        echo "  ptb=$ptb : (not run yet)"
      else
        valid=$(grep -c "seed:.*: [0-9]" "$latest" 2>/dev/null)
        acc=$(grep "Final True Accuracy" "$latest" 2>/dev/null | tail -1)
        if [ -n "$acc" ]; then
          echo "  ptb=$ptb : $acc | valid_seeds: $valid/5 | $(basename "$latest")"
        else
          nan_count=$(grep -c "nan" "$latest" 2>/dev/null)
          echo "  ptb=$ptb : NaN (all 5 seeds) | $(basename "$latest")"
        fi
      fi
    done
    echo ""
  done

  echo ""
  echo "====== Per-Seed Detail (latest run per config) ======"
  for prefix in p1_CEonly_noTau p1_KL_noTau p2_KL0.005 p2_KL0.01 p2_KL0.05 p2_KL0.1 p3_lr0.005 p3_lr0.01; do
    for ptb in 0.0 0.05; do
      latest=$(ls -t "$LOG_DIR/${prefix}_${ptb}_"*.log 2>/dev/null | head -1)
      [ -z "$latest" ] && continue
      echo "--- $(basename $latest) ---"
      grep "seed:" "$latest"
    done
  done
}

# =============================================================================
case "$PHASE" in
  0)
    run_p1_tautune_ablation
    ;;
  1)
    run_p2_kl_scan
    ;;
  2)
    run_p3_lr_scan
    ;;
  results)
    show_results
    ;;
  *)
    echo "Usage: bash $0 {0|1|2|results}"
    echo ""
    echo "  🔴 P1 (0) : τ_tune ablation — disable τ_tune, test CE-only & +tiny KL"
    echo "  🟡 P2 (1) : KL scan — kl ∈ {0.005, 0.01, 0.05, 0.1}"
    echo "  🟢 P3 (2) : prompt_lr scan — lr ∈ {0.005, 0.01}"
    echo "  results   : Show all results summary"
    echo ""
    echo "  推荐跑法："
    echo "    bash $0 0        # 先跑 P1，确认 τ_tune 是否根因 (~2h)"
    echo "    bash $0 results  # 看 P1 结果"
    echo "    bash $0 1        # 如果 P1 确认 τ_tune 根因，跑 KL 扫描 (~3h)"
    echo "    bash $0 2        # prompt_lr 微调 (~3h)"
    exit 1
    ;;
esac
