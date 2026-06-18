#!/bin/bash
# =============================================================================
# 2026-06-17 实验：RobustPrompt-I 稳定性验证 + Filtering Tips 隔离调参
# 服务器：/home/tony/LnL/DFS_HK2 | conda: LnL2 | GPU: RTX 5090
#
# 用法：
#   bash exp_20260617_I_stability_tuning.sh 0        ← 预生成 induced graphs（必须先跑）
#   bash exp_20260617_I_stability_tuning.sh 1        ← Phase 1：稳定性诊断
#   bash exp_20260617_I_stability_tuning.sh 2        ← Phase 2：Filter 隔离调参
#   bash exp_20260617_I_stability_tuning.sh results1 ← 查看 Phase 1 结果
#   bash exp_20260617_I_stability_tuning.sh results2 ← 查看 Phase 2 结果
#
# 所有 job 串行执行（单 GPU，避免 OOM），建议在 tmux/screen 里跑：
#   tmux new -s exp
#   conda activate LnL2
#   bash exp_20260617_I_stability_tuning.sh 0        # Step 0: ~5 min
#   bash exp_20260617_I_stability_tuning.sh 1        # Phase 1: ~2-3 h (8 jobs × ~15-20 min)
#   bash exp_20260617_I_stability_tuning.sh 2        # Phase 2: ~7-9 h (28 jobs × ~15-20 min)
#   Ctrl+B D  (detach tmux)
#
# 静默逻辑：
#   sim_pt 静默：--pt_sim_threshold -1.0    (无节点 csim <= -1.0)
#   degree_pt 静默：--pt_degree_threshold -1  (无节点 deg <= -1)
#   out_detect_pt 静默：--pt_out_detect_threshold -1.0  (无边 cosine <= -1.0)
#   τ_tune 禁用：--pt_threshold -1.1          (所有边 cosine >= -1.1，不剪边)
#
# 日志位置：logs/RobustPrompt-I/
# =============================================================================

PHASE="${1:-1}"

MODEL="./pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth"
LOG_DIR="logs/RobustPrompt-I"
mkdir -p "$LOG_DIR"

# =============================================================================
# Step 0: 预生成 induced graphs（必须先跑，否则并行 job 会竞争写 pickle）
#   只需要每个 attack_method 跑一次，后续 job 直接加载缓存
# =============================================================================
pregenerate_induced() {
  echo "====== Step 0: Pre-generating induced graphs (sequential) ======"

  for ptb in 0.0 0.05; do
    LOG="$LOG_DIR/_gen_induced_${ptb}.log"
    echo "  -> ptb=$ptb  log=$(basename $LOG)"
    CUDA_VISIBLE_DEVICES=0 python MyTask.py \
      --pre_train_model_path "$MODEL" \
      --task NodeTask --dataset_name Cora --preprocess_method none \
      --gnn_type GCN --prompt_type RobustPrompt-I --shot_num 5 --run_split 1 \
      --hid_dim 256 --num_layer 2 --epochs 1 --seed 1 \
      --filter_mode original \
      --pt_sim_threshold 0.2 --pt_degree_threshold 1 --pt_out_detect_threshold 0.4 \
      --prompt_lr 0.001 --pt_threshold 0.25 \
      --weight_mse 0 --weight_kl 0 --weight_constraint 0 \
      --attack_downstream --specified --attack_method Meta_Self-${ptb} \
      > "$LOG" 2>&1
    echo "  <- done (exit code $?)"
  done

  echo "Induced graphs ready. Now run Phase 1."
}

# =============================================================================
# Phase 1: 稳定性诊断（同事四步法）
#   防御 prompt 用默认值（sim=0.2 deg=1 ood=0.4）
# =============================================================================
run_phase1() {
  echo "====== Phase 1: Stability Diagnostics (8 jobs, sequential) ======"
  START_TIME=$(date +%s)

  # Step 1: CE-only (无正则, 无 τ_tune)
  echo "[1/4] CE-only (no reg, no tau_tune)"
  for ptb in 0.0 0.05; do
    LOG="$LOG_DIR/stab1_CEonly_${ptb}_$(date +%m%d_%H%M).log"
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
    echo "  <- done (acc=$(grep 'Final True Accuracy' "$LOG" | tail -1 | grep -oP '[\d.]+'))"
  done

  # Step 2: 只开 τ_tune
  echo "[2/4] +tau_tune (pt=0.25, no reg)"
  for ptb in 0.0 0.05; do
    LOG="$LOG_DIR/stab2_tautune_${ptb}_$(date +%m%d_%H%M).log"
    echo "  -> ptb=$ptb  log=$(basename $LOG)"
    CUDA_VISIBLE_DEVICES=0 python MyTask.py \
      --pre_train_model_path "$MODEL" \
      --task NodeTask --dataset_name Cora --preprocess_method none \
      --gnn_type GCN --prompt_type RobustPrompt-I --shot_num 5 --run_split 1 \
      --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
      --filter_mode original \
      --attack_downstream --specified \
      --prompt_lr 0.001 --pt_threshold 0.25 \
      --weight_mse 0 --weight_kl 0 --weight_constraint 0 \
      --pt_sim_threshold 0.2 --pt_degree_threshold 1 --pt_out_detect_threshold 0.4 \
      --attack_method Meta_Self-${ptb} \
      > "$LOG" 2>&1
    echo "  <- done (acc=$(grep 'Final True Accuracy' "$LOG" | tail -1 | grep -oP '[\d.]+'))"
  done

  # Step 3: + 小 MSE
  echo "[3/4] +small MSE (mse=0.01)"
  for ptb in 0.0 0.05; do
    LOG="$LOG_DIR/stab3_MSE_${ptb}_$(date +%m%d_%H%M).log"
    echo "  -> ptb=$ptb  log=$(basename $LOG)"
    CUDA_VISIBLE_DEVICES=0 python MyTask.py \
      --pre_train_model_path "$MODEL" \
      --task NodeTask --dataset_name Cora --preprocess_method none \
      --gnn_type GCN --prompt_type RobustPrompt-I --shot_num 5 --run_split 1 \
      --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
      --filter_mode original \
      --attack_downstream --specified \
      --prompt_lr 0.001 --pt_threshold 0.25 \
      --weight_mse 0.01 --weight_kl 0 --weight_constraint 0 \
      --pt_sim_threshold 0.2 --pt_degree_threshold 1 --pt_out_detect_threshold 0.4 \
      --attack_method Meta_Self-${ptb} \
      > "$LOG" 2>&1
    echo "  <- done (acc=$(grep 'Final True Accuracy' "$LOG" | tail -1 | grep -oP '[\d.]+'))"
  done

  # Step 4: + 极小 KL
  echo "[4/4] +tiny KL (mse=0.01, kl=0.001)"
  for ptb in 0.0 0.05; do
    LOG="$LOG_DIR/stab4_KL_${ptb}_$(date +%m%d_%H%M).log"
    echo "  -> ptb=$ptb  log=$(basename $LOG)"
    CUDA_VISIBLE_DEVICES=0 python MyTask.py \
      --pre_train_model_path "$MODEL" \
      --task NodeTask --dataset_name Cora --preprocess_method none \
      --gnn_type GCN --prompt_type RobustPrompt-I --shot_num 5 --run_split 1 \
      --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
      --filter_mode original \
      --attack_downstream --specified \
      --prompt_lr 0.001 --pt_threshold 0.25 \
      --weight_mse 0.01 --weight_kl 0.001 --weight_constraint 0 \
      --pt_sim_threshold 0.2 --pt_degree_threshold 1 --pt_out_detect_threshold 0.4 \
      --attack_method Meta_Self-${ptb} \
      > "$LOG" 2>&1
    echo "  <- done (acc=$(grep 'Final True Accuracy' "$LOG" | tail -1 | grep -oP '[\d.]+'))"
  done

  ELAPSED=$(( $(date +%s) - START_TIME ))
  echo "Phase 1 done in ${ELAPSED}s. Check: bash $0 results1"
}

# =============================================================================
# Phase 2: Filtering Tips 隔离调参
#   前提：Phase 1 无 NaN。调一个 filter 时其余两个静默。
#   基础配置来自 Phase 1 最稳组合（默认 Step 3: CE+小MSE）
# =============================================================================
run_phase2() {
  echo "====== Phase 2: Isolated Filter Tuning (28 jobs, sequential) ======"
  START_TIME=$(date +%s)

  STABLE_LR=0.001
  STABLE_PT=0.25
  STABLE_MSE=0.01
  STABLE_KL=0

  # Phase 2a: sim_pt 独立（静默 degree=-1, ood=-1.0）
  echo "[2a] sim_pt: 0.2 0.3 0.4 0.5 0.6  x  clean/0.05"
  for sim_t in 0.2 0.3 0.4 0.5 0.6; do
    for ptb in 0.0 0.05; do
      LOG="$LOG_DIR/ft_sim${sim_t}_${ptb}_$(date +%m%d_%H%M).log"
      echo "  -> sim=$sim_t ptb=$ptb  log=$(basename $LOG)"
      CUDA_VISIBLE_DEVICES=0 python MyTask.py \
        --pre_train_model_path "$MODEL" \
        --task NodeTask --dataset_name Cora --preprocess_method none \
        --gnn_type GCN --prompt_type RobustPrompt-I --shot_num 5 --run_split 1 \
        --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
        --filter_mode original \
        --attack_downstream --specified \
        --prompt_lr $STABLE_LR --pt_threshold $STABLE_PT \
        --weight_mse $STABLE_MSE --weight_kl $STABLE_KL --weight_constraint 0 \
        --pt_sim_threshold ${sim_t} \
        --pt_degree_threshold -1 --pt_out_detect_threshold -1.0 \
        --attack_method Meta_Self-${ptb} \
        > "$LOG" 2>&1
      echo "  <- done (acc=$(grep 'Final True Accuracy' "$LOG" | tail -1 | grep -oP '[\d.]+'))"
    done
  done

  # Phase 2b: degree_pt 独立（静默 sim=-1.0, ood=-1.0）
  echo "[2b] degree_pt: 1 2 3 5  x  clean/0.05"
  for deg_t in 1 2 3 5; do
    for ptb in 0.0 0.05; do
      LOG="$LOG_DIR/ft_deg${deg_t}_${ptb}_$(date +%m%d_%H%M).log"
      echo "  -> deg=$deg_t ptb=$ptb  log=$(basename $LOG)"
      CUDA_VISIBLE_DEVICES=0 python MyTask.py \
        --pre_train_model_path "$MODEL" \
        --task NodeTask --dataset_name Cora --preprocess_method none \
        --gnn_type GCN --prompt_type RobustPrompt-I --shot_num 5 --run_split 1 \
        --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
        --filter_mode original \
        --attack_downstream --specified \
        --prompt_lr $STABLE_LR --pt_threshold $STABLE_PT \
        --weight_mse $STABLE_MSE --weight_kl $STABLE_KL --weight_constraint 0 \
        --pt_sim_threshold -1.0 --pt_out_detect_threshold -1.0 \
        --pt_degree_threshold ${deg_t} \
        --attack_method Meta_Self-${ptb} \
        > "$LOG" 2>&1
      echo "  <- done (acc=$(grep 'Final True Accuracy' "$LOG" | tail -1 | grep -oP '[\d.]+'))"
    done
  done

  # Phase 2c: out_detect_pt 独立（静默 sim=-1.0, degree=-1）
  echo "[2c] out_detect_pt: 0.3 0.4 0.5 0.6 0.7  x  clean/0.05"
  for ood_t in 0.3 0.4 0.5 0.6 0.7; do
    for ptb in 0.0 0.05; do
      LOG="$LOG_DIR/ft_ood${ood_t}_${ptb}_$(date +%m%d_%H%M).log"
      echo "  -> ood=$ood_t ptb=$ptb  log=$(basename $LOG)"
      CUDA_VISIBLE_DEVICES=0 python MyTask.py \
        --pre_train_model_path "$MODEL" \
        --task NodeTask --dataset_name Cora --preprocess_method none \
        --gnn_type GCN --prompt_type RobustPrompt-I --shot_num 5 --run_split 1 \
        --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
        --filter_mode original \
        --attack_downstream --specified \
        --prompt_lr $STABLE_LR --pt_threshold $STABLE_PT \
        --weight_mse $STABLE_MSE --weight_kl $STABLE_KL --weight_constraint 0 \
        --pt_sim_threshold -1.0 --pt_degree_threshold -1 \
        --pt_out_detect_threshold ${ood_t} \
        --attack_method Meta_Self-${ptb} \
        > "$LOG" 2>&1
      echo "  <- done (acc=$(grep 'Final True Accuracy' "$LOG" | tail -1 | grep -oP '[\d.]+'))"
    done
  done

  ELAPSED=$(( $(date +%s) - START_TIME ))
  echo "Phase 2 done in ${ELAPSED}s. Check: bash $0 results2"
}
# =============================================================================
# 结果提取
# =============================================================================
show_results1() {
  echo "====== Phase 1 Results ======"
  for step in stab1_CEonly stab2_tautune stab3_MSE stab4_KL; do
    echo "--- $step ---"
    for ptb in 0.0 0.05; do
      latest=$(ls -t "$LOG_DIR/${step}_${ptb}_"*.log 2>/dev/null | head -1)
      if [ -z "$latest" ]; then
        echo "  ptb=$ptb : (no log yet)"
      else
        acc=$(grep "Final True Accuracy" "$latest" | tail -1 | grep -oP '[\d.]+')
        echo "  ptb=$ptb : acc=$acc  |  $(basename "$latest")"
      fi
    done
    echo ""
  done
}

show_results2() {
  echo "====== Phase 2a: sim_pt ======"
  for sim_t in 0.2 0.3 0.4 0.5 0.6; do
    printf "sim=%-4s" "$sim_t"
    for ptb in 0.0 0.05; do
      latest=$(ls -t "$LOG_DIR/ft_sim${sim_t}_${ptb}_"*.log 2>/dev/null | head -1)
      [ -z "$latest" ] && printf "  %-5s: ------" "$ptb" || {
        acc=$(grep "Final True Accuracy" "$latest" | tail -1 | grep -oP '[\d.]+')
        printf "  %-5s: %-8s" "$ptb" "$acc"
      }
    done
    echo ""
  done

  echo ""
  echo "====== Phase 2b: degree_pt ======"
  for deg_t in 1 2 3 5; do
    printf "deg=%-4s" "$deg_t"
    for ptb in 0.0 0.05; do
      latest=$(ls -t "$LOG_DIR/ft_deg${deg_t}_${ptb}_"*.log 2>/dev/null | head -1)
      [ -z "$latest" ] && printf "  %-5s: ------" "$ptb" || {
        acc=$(grep "Final True Accuracy" "$latest" | tail -1 | grep -oP '[\d.]+')
        printf "  %-5s: %-8s" "$ptb" "$acc"
      }
    done
    echo ""
  done

  echo ""
  echo "====== Phase 2c: out_detect_pt ======"
  for ood_t in 0.3 0.4 0.5 0.6 0.7; do
    printf "ood=%-4s" "$ood_t"
    for ptb in 0.0 0.05; do
      latest=$(ls -t "$LOG_DIR/ft_ood${ood_t}_${ptb}_"*.log 2>/dev/null | head -1)
      [ -z "$latest" ] && printf "  %-5s: ------" "$ptb" || {
        acc=$(grep "Final True Accuracy" "$latest" | tail -1 | grep -oP '[\d.]+')
        printf "  %-5s: %-8s" "$ptb" "$acc"
      }
    done
    echo ""
  done
}

# =============================================================================
case "$PHASE" in
  0)
    pregenerate_induced
    ;;
  1)
    run_phase1
    ;;
  2)
    run_phase2
    ;;
  results1)
    show_results1
    ;;
  results2)
    show_results2
    ;;
  *)
    echo "Usage: bash $0 {0|1|2|results1|results2}"
    echo ""
    echo "  0        Pre-generate induced graph pickles (MUST run first, once)"
    echo "  1        Phase 1: stability diagnostics (8 jobs)"
    echo "  2        Phase 2: isolated filter tuning (28 jobs)"
    echo "  results1 Show Phase 1 results"
    echo "  results2 Show Phase 2 results"
    echo ""
    echo "  Recommended order:"
    echo "    bash $0 0         # once: generate induced graphs"
    echo "    bash $0 1         # stability tests"
    echo "    bash $0 results1  # check no NaN"
    echo "    bash $0 2         # filter tuning"
    echo "    bash $0 results2  # check results"
    exit 1
    ;;
esac
