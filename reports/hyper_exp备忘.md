实验矩阵（910 runs）：

Section 1 — GPPT Baseline                    60 runs
  stable BB × 6 ptb × 5 seeds
  peak   BB × 6 ptb × 5 seeds

Section 2 — GCL Linear Probe (attacked)      10 runs
  stable BB × 5 ptb (0.05-0.25)
  peak   BB × 5 ptb (0.05-0.25)

Section 3a — sim_pt sweep                   300 runs
  2 BB × 5 values {0.2,0.3,0.4,0.5,0.6} × 6 ptb × 5 seeds

Section 3b — degree_pt sweep                240 runs
  2 BB × 4 values {1,2,3,5} × 6 ptb × 5 seeds

Section 3c — out_detect_pt sweep            300 runs
  2 BB × 5 values {0.3,0.4,0.5,0.6,0.7} × 6 ptb × 5 seeds
固定基础参数（Meeting 13 optimal）：


prompt_lr=0.01, pt_threshold=0.25, weight_mse=0.0, weight_kl=0.001
两个 Backbone：

标签	路径	来源
stable	permE/dropN/lr=0.001/ratio=0.2/seed=1	组均值 0.5424
peak	permE/maskN/lr=0.001/ratio=0.3/seed=1	单点 0.6262
日志目录：


logs/baselines/           — GPPT + GCL LP
logs/single_filter_sim/   — sim_pt sweep
logs/single_filter_degree/ — degree_pt sweep
logs/single_filter_ood/   — out_detect_pt sweep
3. 服务器执行



# 跑全量实验矩阵（后台，写 nohup.out）
nohup bash run_all_experiments.sh &
