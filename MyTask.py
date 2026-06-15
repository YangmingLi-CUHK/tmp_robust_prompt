import numpy as np
from prompt_graph.tasker import NodeTask, GraphTask
from prompt_graph.utils import seed_everything
from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args
# 非 Cora 链路已删除: load4graph, load4link, induced_graphs_from_edges, CustomTUDataset
# 如需 GraphTask/LinkTask，请从 git history 恢复相关函数
import torch

args = get_args()


if args.task == 'NodeTask':
    # 两种统计方式，按照split划分和按照seed划分
    # 按照split划分
    # 记录每个split在不同seed的值 初始化一个dict
    all_split_acc_list = {}
    all_split_acc_dict = {}
    for split_num in args.run_split:
        all_split_acc_dict[split_num] = {}
        all_split_acc_list[split_num] = []

    # 按照seed划分
    all_seed_acc_list  = {}
    all_seed_acc_dict  = {}

    for seed in args.seed:
        seed_everything(seed)
        # 对多次划分进行测试
        # 记录seed对应不同的split
        seed_acc_list = []
        seed_acc_dict = {}
        for split_num in args.run_split:
            tasker = NodeTask(pre_train_model_path = args.pre_train_model_path, hid_dim=args.hid_dim,
                            dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type,
                            prompt_type = args.prompt_type, epochs = args.epochs, shot_num = args.shot_num, run_split= split_num, preprocess_method = args.preprocess_method, attack_downstream = args.attack_downstream, attack_method = args.attack_method, specified = args.specified, adaptive = args.adaptive, adaptive_scenario=args.adaptive_scenario, adaptive_split= args.adaptive_split, adaptive_attack_model= args.adaptive_attack_model, adaptive_ptb_rate= args.adaptive_ptb_rate, filter_mode=args.filter_mode, filter_sim1_weight=args.filter_sim1_weight, filter_sim2_weight=args.filter_sim2_weight, filter_hybrid_alpha=args.filter_hybrid_alpha,
                            pt_threshold=args.pt_threshold, weight_mse=args.weight_mse, weight_kl=args.weight_kl, weight_constraint=args.weight_constraint, temperature=args.temperature, pt_sim_threshold=args.pt_sim_threshold, pt_degree_threshold=args.pt_degree_threshold, pt_out_detect_threshold=args.pt_out_detect_threshold, p_plus=args.p_plus, use_attention=args.use_attention, cosine_constraint=args.cosine_constraint, prompt_lr=args.prompt_lr)
            
            test_acc = tasker.run()

            # 记录每个split在不同seed的值
            all_split_acc_list[split_num].append(test_acc)
            all_split_acc_dict[split_num][seed] = test_acc

            # 记录同一seed下对应不同的split
            seed_acc_list.append(test_acc)
            seed_acc_dict[split_num] = test_acc

        all_seed_acc_list[seed] = seed_acc_list
        all_seed_acc_dict[seed] = seed_acc_dict


    print(all_seed_acc_dict)
    print(all_split_acc_dict)
    print('########################################################################################')
    # 打印一个seed下多个split的平均
    for seed, seed_acc_dict in all_seed_acc_dict.items():
        for split_num, acc in seed_acc_dict.items():
            print('seed: {} | split {} : {}'.format(seed, split_num, acc))

        seed_final_acc, seed_final_acc_std = np.mean(all_seed_acc_list[seed]), np.std(all_seed_acc_list[seed])
        print(f"# Seed {seed} Muti Split Final Acc: {seed_final_acc:.4f}±{seed_final_acc_std:.4f}")
    print('########################################################################################')
    # 打印一个split下多个seed的平均
    for split_num, split_acc_dict in all_split_acc_dict.items():
        # 过滤掉 NaN 值（loss 为 NaN 的 seed 会返回 NaN）
        valid_accs = [a for a in all_split_acc_list[split_num] if not (isinstance(a, float) and np.isnan(a))]
        if len(valid_accs) == 0:
            print(f"Split {split_num}: All seeds returned NaN, no valid results.")
            continue
        if len(valid_accs) == 1:
            print("There's only one valid result, it's recommended to try several seeds.")
            trimmed_accs = valid_accs
        else:
            # 对所有seed的结果排序，去掉最低和最高的值再求平均
            valid_accs.sort(reverse=True)
            trimmed_accs = valid_accs[1:-1] if len(valid_accs) > 2 else valid_accs
        for seed, acc in split_acc_dict.items():
            print('split: {} | seed {} : {}'.format(split_num, seed, acc))

        split_final_acc, split_final_acc_std = np.mean(trimmed_accs), np.std(trimmed_accs)
        print(f"# Split {split_num} Muti Seed Acc (trimmed, {len(trimmed_accs)}/{len(valid_accs)} seeds): {split_final_acc:.4f}±{split_final_acc_std:.4f}")
    print('########################################################################################')








# GraphTask / LinkTask 已暂时禁用：相关数据加载函数 (load4graph, load4link,
# induced_graphs_from_edges, CustomTUDataset) 已从 load4data.py 删除。
# 如需恢复，请从 git history 找回。

