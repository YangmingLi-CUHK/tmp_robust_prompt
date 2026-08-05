import numpy as np
from prompt_graph.tasker import NodeTask  # GraphTask removed (2026-06-16)
from prompt_graph.utils import seed_everything
from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import get_args
# GraphTask/LinkTask 依赖的旧函数已于 2026-06-16 移除，仅保留 NodeTask
# from prompt_graph.data import load4graph, load4link, induced_graphs_from_edges, CustomTUDataset
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
    all_seed_acc_list = {}
    all_seed_acc_dict = {}

    for seed in args.seed:
        seed_everything(seed)
        # 对多次划分进行测试
        # 记录seed对应不同的split
        seed_acc_list = []
        seed_acc_dict = {}
        for split_num in args.run_split:
            tasker = NodeTask(
                pre_train_model_path=args.pre_train_model_path,
                pretrain_dataset_name=args.pretrain_dataset_name,
                hid_dim=args.hid_dim,
                dataset_name=args.dataset_name,
                num_layer=args.num_layer,
                gnn_type=args.gnn_type,
                prompt_type=args.prompt_type,
                epochs=args.epochs,
                shot_num=args.shot_num,
                run_split=split_num,
                preprocess_method=args.preprocess_method,
                svd_out_dim=args.svd_out_dim,
                downstream_svd_cache=args.downstream_svd_cache,
                attack_downstream=args.attack_downstream,
                attack_method=args.attack_method,
                specified=args.specified,
                strict_attack_raw=args.strict_attack_raw,
                adaptive=args.adaptive,
                adaptive_scenario=args.adaptive_scenario,
                adaptive_split=args.adaptive_split,
                adaptive_attack_model=args.adaptive_attack_model,
                adaptive_ptb_rate=args.adaptive_ptb_rate,
                filter_mode=args.filter_mode,
                filter_sim1_weight=args.filter_sim1_weight,
                filter_sim2_weight=args.filter_sim2_weight,
                filter_hybrid_alpha=args.filter_hybrid_alpha,
                filter_lp_hidden_dim=args.filter_lp_hidden_dim,
                filter_lp_epochs=args.filter_lp_epochs,
                filter_lp_lr=args.filter_lp_lr,
                filter_lp_neg_ratio=args.filter_lp_neg_ratio,
                filter_lp_threshold_mode=args.filter_lp_threshold_mode,
                filter_lp_max_train_pairs=args.filter_lp_max_train_pairs,
                filter_lp_pca_dim=args.filter_lp_pca_dim,
                pt_threshold=args.pt_threshold,
                weight_mse=args.weight_mse,
                weight_kl=args.weight_kl,
                weight_constraint=args.weight_constraint,
                temperature=args.temperature,
                pt_sim_threshold=args.pt_sim_threshold,
                pt_degree_threshold=args.pt_degree_threshold,
                pt_out_detect_threshold=args.pt_out_detect_threshold,
                pt_nsp_threshold=args.pt_nsp_threshold,
                pt_focusedcleaner_threshold=args.pt_focusedcleaner_threshold,
                nsp_order=args.nsp_order,
                p_plus=args.p_plus,
                use_attention=args.use_attention,
                cosine_constraint=args.cosine_constraint,
                prompt_lr=args.prompt_lr,
                prompt_variant=args.prompt_variant,
                device=args.device,
            )

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
        if len(all_split_acc_list[split_num]) == 1:
            print("There's only one result, it's recommended to try several seeds.")
        for seed, acc in split_acc_dict.items():
            print('split: {} | seed {} : {}'.format(split_num, seed, acc))

        split_final_acc, split_final_acc_std = np.mean(all_split_acc_list[split_num]), np.std(all_split_acc_list[split_num])
        print(f"# Split {split_num} Muti Seed Acc (all seeds): {split_final_acc:.4f}±{split_final_acc_std:.4f}")
    print('########################################################################################')


# =============================================================================
# GraphTask / LinkTask 已于 2026-06-16 移除（依赖的 load4graph/load4link 等函数已删除）。
# 如需恢复，请从 git history 找回 load4data.py 中的旧函数 + 取消下面注释。
# =============================================================================
# elif args.task == 'GraphTask':
#     for seed in args.seed:
#         seed_everything(seed)
#         input_dim, output_dim, dataset = load4graph(args.dataset_name)
#         tasker = GraphTask(pre_train_model_path=args.pre_train_model_path,
#                         dataset_name=args.dataset_name, num_layer=args.num_layer, gnn_type=args.gnn_type, hid_dim=args.hid_dim, prompt_type=args.prompt_type, epochs=args.epochs,
#                         shot_num=args.shot_num, device=args.device, lr=args.lr, wd=args.decay,
#                         batch_size=args.batch_size, dataset=dataset, input_dim=input_dim, output_dim=output_dim, task_type='GraphTask', filter_mode=args.filter_mode, filter_sim1_weight=args.filter_sim1_weight, filter_sim2_weight=args.filter_sim2_weight, filter_hybrid_alpha=args.filter_hybrid_alpha)
#         _, test_acc, std_test_acc, f1, std_f1, roc, std_roc, _, _ = tasker.run()
#
# elif args.task == 'LinkTask':
#     for seed in args.seed:
#         seed_everything(seed)
#         assert args.dataset_name in ['Cora', 'Citeseer', 'PubMed', 'Wisconsin', 'ogbn-arxiv']
#         dataset = load4link(args.dataset_name)
#         data = dataset[0]
#         if args.dataset_name == 'ogbn-arxiv':
#             data.y = data.y.squeeze()
#         input_dim = dataset.num_features
#         out_dim = dataset.num_classes
#         dataset = induced_graphs_from_edges(data, args.device, smallest_size=1, largest_size=30)
#         print("num edge subgraphs: ", len(dataset))
#         dataset = CustomTUDataset(dataset)
#         tasker = GraphTask(pre_train_model_path=args.pre_train_model_path,
#                     dataset_name=args.dataset_name, num_layer=args.num_layer, gnn_type=args.gnn_type, hid_dim=args.hid_dim, prompt_type=args.prompt_type, epochs=args.epochs,
#                     shot_num=args.shot_num, device=args.device, lr=args.lr, wd=args.decay,
#                     batch_size=1024, dataset=dataset, input_dim=input_dim, output_dim=2, task_type='LinkTask', filter_mode=args.filter_mode, filter_sim1_weight=args.filter_sim1_weight, filter_sim2_weight=args.filter_sim2_weight, filter_hybrid_alpha=args.filter_hybrid_alpha)
#         _, test_acc, std_test_acc, f1, std_f1, roc, std_roc, _, _ = tasker.run()
