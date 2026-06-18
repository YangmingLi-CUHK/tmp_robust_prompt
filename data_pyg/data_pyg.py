"""
统一数据获取函数（仅保留攻击数据分支）。

非 Cora 数据集分支（Planetoid, CitationFull, Coauthor, Amazon, Reddit, WikiCS,
Flickr, Yelp, PPI, ogbn）已删除。如需恢复，请从 git history 找回。
"""
from data_pyg.Attack_data.attackdata               import AttackDataset
from data_pyg.Attack_unit_test_data.attackunitdata import AttackUnitDataset


def get_dataset(path, name, attackmethod=None, attackptb=None, adaptive_dict=None):
    """获取攻击数据集。

    Parameters
    ----------
    path : str
        数据根路径（Attack_data 或 Attack_unit_test_data 的路径）。
    name : str
        数据集名，必须以 'Attack-' 或 'Unit-' 开头。
        例如: 'Attack-Cora', 'Unit-Citeseer'。
    attackmethod : str, optional
        攻击方法名（Meta_Self 等），仅 Attack- 分支需要。
    attackptb : str or float, optional
        扰动率，仅 Attack- 分支需要。
    adaptive_dict : EasyDict, optional
        自适应攻击参数字典，仅 Unit- 分支需要。

    Returns
    -------
    InMemoryDataset
    """
    if name.startswith('Attack'):
        return AttackDataset(
            root=path, name=name,
            attackmethod=attackmethod,
            ptb_rate=attackptb
        )

    if name.startswith('Unit'):
        return AttackUnitDataset(
            root=path, name=name,
            scenario=adaptive_dict.scenario,
            split=adaptive_dict.split,
            adaptive_attack_model=adaptive_dict.adaptive_attack_model,
            ptb_rate=adaptive_dict.ptb_rate
        )

    raise ValueError(
        f"Unsupported dataset name: {name!r}. "
        f"Currently only 'Attack-*' and 'Unit-*' datasets are supported. "
        f"Non-Cora dataset branches were removed — restore from git history if needed."
    )
