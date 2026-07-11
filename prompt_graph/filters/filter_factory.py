from .neighbor_similarity_filter import HybridFilter, NeighborSimilarityFilter, OriginalFilter
from .nsp_filter import NSPFilter
from .focusedcleaner_lp_filter import FocusedCleanerLPFilter


def build_filter(args):
    filter_mode = getattr(args, "filter_mode", "original")
    threshold = getattr(args, "pt_threshold", 0.0)
    sim1_weight = getattr(args, "filter_sim1_weight", 0.5)
    sim2_weight = getattr(args, "filter_sim2_weight", 0.5)
    hybrid_alpha = getattr(args, "filter_hybrid_alpha", 0.5)
    nsp_order = getattr(args, "nsp_order", 2)
    lp_hidden_dim = getattr(args, "filter_lp_hidden_dim", 0)
    lp_epochs = getattr(args, "filter_lp_epochs", 50)
    lp_lr = getattr(args, "filter_lp_lr", 0.1)
    lp_neg_ratio = getattr(args, "filter_lp_neg_ratio", 1.0)
    lp_threshold_mode = getattr(args, "filter_lp_threshold_mode", "gmean")
    lp_max_train_pairs = getattr(args, "filter_lp_max_train_pairs", 200000)
    lp_pca_dim = getattr(args, "filter_lp_pca_dim", 0)

    if filter_mode == "original":
        return OriginalFilter(threshold=threshold)
    if filter_mode == "neighbor_similarity":
        return NeighborSimilarityFilter(threshold=threshold, w1=sim1_weight, w2=sim2_weight)
    if filter_mode == "hybrid":
        return HybridFilter(
            threshold=threshold,
            w1=sim1_weight,
            w2=sim2_weight,
            alpha=hybrid_alpha,
        )
    if filter_mode == "nsp":
        return NSPFilter(threshold=threshold, order=nsp_order)
    if filter_mode == "focusedcleaner_lp":
        return FocusedCleanerLPFilter(
            threshold=threshold,
            hidden_dim=lp_hidden_dim,
            epochs=lp_epochs,
            lr=lp_lr,
            neg_ratio=lp_neg_ratio,
            threshold_mode=lp_threshold_mode,
            max_train_pairs=lp_max_train_pairs,
            pca_dim=lp_pca_dim,
        )
    raise ValueError(f"Unsupported filter_mode: {filter_mode}")
