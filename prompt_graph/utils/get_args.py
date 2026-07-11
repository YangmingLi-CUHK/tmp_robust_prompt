import argparse


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--task', type=str)
    parser.add_argument('--dataset_name', type=str, default='Cora', help='Choose the dataset of pretrainor downstream task')
    parser.add_argument('--device', type=int, default=0,
                        help='Which gpu to use if any (default: 0)')
    parser.add_argument('--gnn_type', type=str, default="GCN",
                        help='We support gnn like \GCN\ \GAT\ \GT\ \GCov\ \GIN\ \GraphSAGE\, please read ProG.model module')
    parser.add_argument('--prompt_type', type=str, default="All-in-one",
                        help='Choose the prompt type for node or graph task, for node task,we support \GPPT\, \All-in-one\, \Gprompt\ for graph task , \All-in-one\, \Gprompt\, \GPF\, \GPF-plus\ ')
    parser.add_argument('--hid_dim', type=int, default=128,
                        help='hideen layer of GNN dimensions (default: 300)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train (default: 50)')
    parser.add_argument('--shot_num', type=int, default=5, help='Number of shots')
    parser.add_argument('--pre_train_model_path', type=str, default='None',
                        help='add pre_train_model_path to the downstream task, the model is self-supervise model if the path is None and prompttype is None.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='Weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='Number of GNN message passing layers (default: 3).')

    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='Dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='Graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='How the node features across layers are combined. last, sum, max or concat')

    # parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting dataset.")
    parser.add_argument('--seed', nargs='+', type=int, help="Seed for splitting dataset.")
    parser.add_argument('--run_split', nargs='+', type=int, help="run split num")
    # parser.add_argument('--run_split', type=int, help= "run split num")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for running experiments.")
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataset loading')
    parser.add_argument('--num_layers', type=int, default=1, help='A range of [1,2,3]-layer MLPs with equal width')
    parser.add_argument('--pnum', type=int, default=5, help='The number of independent basis for GPF-plus')

    # add ssh
    parser.add_argument('--preprocess_method', type=str, default='None', help='Choose preprocess method svd')
    parser.add_argument('--attack_downstream', action='store_true', default=False, help='Attack Downstream Task')
    parser.add_argument('--specified', action='store_true', default=False,
                        help='Attack specified split, Used for some distribution-based attacks')
    # ArgumentParser在传布尔类型变量时，传入参数按字符串处理，所以无论传入什么值，参数值都为True。
    parser.add_argument('--attack_method', type=str, default='None')  # ['DICE-0.1','Meta_Self-0.05' ,...] 攻击方式-扰动率
    # 如果用自适应攻击，就使用下面的参数
    parser.add_argument('--adaptive', action='store_true', default=False, help='Unit Test')
    parser.add_argument('--adaptive_scenario', type=str, default='None')  # 'poisoning'
    parser.add_argument('--adaptive_split', type=int, default=0)
    parser.add_argument('--adaptive_attack_model', type=str, default='None')
    parser.add_argument('--adaptive_ptb_rate', type=float, default=0.)

    parser.add_argument('--filter_mode', type=str, default='original',
                        choices=['original', 'neighbor_similarity', 'hybrid', 'focusedcleaner_lp'],
                        help='Choose which robust filter to use. Default keeps the current behavior unchanged.')
    parser.add_argument('--filter_sim1_weight', type=float, default=0.5,
                        help='Weight for cosine similarity on H1 = A @ X in neighbor-similarity filtering.')
    parser.add_argument('--filter_sim2_weight', type=float, default=0.5,
                        help='Weight for cosine similarity on H2 = A @ A @ X in neighbor-similarity filtering.')
    parser.add_argument('--filter_hybrid_alpha', type=float, default=0.5,
                        help='Weight for original filter signal in hybrid mode.')

    # FocusedCleaner-LP filter parameters
    parser.add_argument('--filter_lp_hidden_dim', type=int, default=0,
                        help='Hidden dimension for FocusedCleaner-LP. 0 means use the input feature dimension.')
    parser.add_argument('--filter_lp_epochs', type=int, default=50,
                        help='Training epochs for FocusedCleaner-LP link predictor.')
    parser.add_argument('--filter_lp_lr', type=float, default=0.1,
                        help='Learning rate for FocusedCleaner-LP link predictor.')
    parser.add_argument('--filter_lp_neg_ratio', type=float, default=1.0,
                        help='Negative non-edge samples per positive edge for FocusedCleaner-LP.')
    parser.add_argument('--filter_lp_threshold_mode', type=str, default='gmean',
                        choices=['gmean', 'fixed'],
                        help='FocusedCleaner-LP threshold: gmean follows the paper; fixed uses pt_threshold.')
    parser.add_argument('--filter_lp_max_train_pairs', type=int, default=200000,
                        help='Maximum positive+negative training pairs for FocusedCleaner-LP.')
    parser.add_argument('--filter_lp_pca_dim', type=int, default=-1,
                        help='PCA attribute dimension for FocusedCleaner-LP. -1 uses the number of classes.')

    # RobustPrompt-T 超参数（GPromptShield）
    parser.add_argument('--pt_threshold', type=float, default=0.5,
                        help='τ_tune: cosine similarity threshold for edge pruning in Tune()')
    parser.add_argument('--weight_mse', type=float, default=0.1,
                        help='Weight for MSE smoothness loss (L_s)')
    parser.add_argument('--weight_kl', type=float, default=0.1,
                        help='Weight for KL distribution alignment loss')
    parser.add_argument('--weight_constraint', type=float, default=0.2,
                        help='Weight for prompt orthogonality constraint loss')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for KL divergence in loss_pt')
    parser.add_argument('--pt_sim_threshold', type=float, default=0.2,
                        help='Cosine similarity threshold for sim_pt node selection '
                             '(best clean 0.4398@0.2; higher=more nodes get sim_pt)')
    parser.add_argument('--pt_degree_threshold', type=int, default=1,
                        help='Degree threshold for degree_pt node selection '
                             '(deg=1 best clean 0.4435 & attacked 0.2448; higher=more nodes)')
    parser.add_argument('--pt_out_detect_threshold', type=float, default=0.4,
                        help='Cosine similarity threshold for out_detect_pt OOD edge detection '
                             '(ood=0.4 most robust: clean 0.4490 attacked 0.2485; smallest gap)')
    parser.add_argument('--pt_nsp_threshold', type=float, default=0.3,
                        help='Neighbor-similarity threshold for nsp_pt (RobustPrompt-T-NSP): an edge whose '
                             'endpoints have (A^order X) cosine <= this is flagged; both endpoints become anomalous nodes')
    parser.add_argument('--nsp_order', type=int, default=2,
                        help='Aggregation order for NSP neighbor embeddings N = (A^order) X (default 2)')
    parser.add_argument('--pt_focusedcleaner_threshold', type=float, default=0.5,
                        help='FocusedCleaner-LP threshold for focusedcleaner_pt (RobustPrompt-T-NSP): edges with link '
                             'probability below this are flagged; both endpoints become anomalous nodes')
    parser.add_argument('--p_plus', dest='p_plus', action='store_true', default=True,
                        help='Use p_plus mode (20-token bank + learned combination)')
    parser.add_argument('--no_p_plus', dest='p_plus', action='store_false',
                        help='Disable p_plus mode, use single shared prompt per type')
    parser.add_argument('--use_attention', dest='use_attention', action='store_true', default=False,
                        help='Enable Self-Attention fusion of multi-defense prompts '
                             '(off by default: attention found harmful in experiments)')
    parser.add_argument('--no_attention', dest='use_attention', action='store_false',
                        help='Disable attention fusion, use averaging instead (default)')
    parser.add_argument('--cosine_constraint', dest='cosine_constraint', action='store_true', default=True,
                        help='Enable cosine-based prompt orthogonality constraint')
    parser.add_argument('--no_cosine_constraint', dest='cosine_constraint', action='store_false',
                        help='Disable cosine constraint on prompts')
    parser.add_argument('--prompt_lr', type=float, default=0.01,
                        help='Learning rate for RobustPrompt-T optimizer (prompt + answering head)')

    # GraphCL 预训练专用的数据增强参数
    parser.add_argument('--aug1', type=str, default='dropN', choices=['dropN', 'permE', 'maskN'],
                        help='GraphCL augmentation method 1')
    parser.add_argument('--aug2', type=str, default='permE', choices=['dropN', 'permE', 'maskN'],
                        help='GraphCL augmentation method 2')
    parser.add_argument('--aug_ratio', type=float, default=None,
                        help='GraphCL augmentation ratio (0.0-1.0). If not set, randomly chosen from {0.1, 0.2, 0.3} per run.')

    # GraphMAE 预训练专用参数
    parser.add_argument('--mask_rate', type=float, default=0.75, help='GraphMAE node masking ratio (0.0-1.0)')
    parser.add_argument('--drop_edge_rate', type=float, default=0.0, help='GraphMAE edge dropping ratio (0.0-1.0)')
    parser.add_argument('--replace_rate', type=float, default=0.1, help='GraphMAE noise replacement ratio (0.0-1.0)')
    parser.add_argument('--loss_fn', type=str, default='sce', choices=['sce', 'mse'],
                        help='GraphMAE reconstruction loss function')
    parser.add_argument('--alpha_l', type=float, default=2.0, help='GraphMAE SCE loss alpha')

    # RobustPrompt-T 代码版本选择
    parser.add_argument('--prompt_variant', type=str, default='ours', choices=['ours', 'original'],
                        help='RobustPrompt-T variant: "ours" (modified) or "original" (paper code, with pass/假attention/特征级τ_tune)')

    args = parser.parse_args()
    return args
