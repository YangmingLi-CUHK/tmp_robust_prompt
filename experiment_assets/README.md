# 实验资产

本目录只保存需要随仓库发布、供跨机器复现或复用的共享checkpoint及最小选择证据。训练过程仍可把新输出写入`pre_trained_model/`和`pre_trained_model_raw/`；日志、污染图、few-shot split及可重算SVD缓存不复制到这里。

## 目录

- `citeseer_svd100_transductive/`：当前180次Transductive主线使用的Peak/Stable Citeseer-SVD100 backbone。
- `citeseer_svd100_transductive/selection_evidence/`：135次clean评估的完整seed级结果、组汇总、运行manifest和source-SVD收据快照。
- `cora_graphcl_meeting14_legacy/`：Meeting14阶段保留的2份Cora-GraphCL backbone。
- `cora_graphmae_legacy/`：历史Cora-GraphMAE64 backbone。
- `manifest.tsv`：共享checkpoint的当前路径、历史记录路径、SHA256、维数及选择口径。

## 口径边界

- 当前Peak/Stable按validation选择；控制器还会校验文件SHA256、state_dict和`100→256→256`维数。
- Meeting14旧Peak/Stable均属于test-informed历史选择，只能标作legacy，不能与当前validation口径混称。
- GraphMAE64缺少完整选择收据，只保留为provenance-limited历史资产。
- `AttrPrompt-main/**/save*`中的模型与其相对加载路径绑定，继续原位保存；移动会破坏teacher加载语义。
- `reference/code_sources/`中的权重属于上游代码快照，不视为本项目共享资产。
- `data_attack_fewshot/`中的污染图和split是数据输入，继续由现有loader及哈希审计管理。
- `data/preprocessed/`和`data/deeprobust/`中的SVD文件是可重算缓存，不进入Git发布资产。

## 原位资产索引

| 路径 | 数量/类型 | 状态 | 原位原因 |
|---|---:|---|---|
| `AttrPrompt-main/AttrPrompt-main/save*/` | 53个`.pth/.pt` | method-local | 训练器把这些相对目录同时作为teacher输入和运行输出；继续由AttrPrompt自身管理。 |
| `reference/code_sources/**/pre_trained_gnn/` | 8个`.pth` | upstream-snapshot | 属于官方/上游代码快照，不冒充本仓实验产物。 |
| `data_attack_fewshot/Cora/shot_5/1/` | corrected-budget污染图、特征、标签和split | runtime-data | 当前loader与控制器按固定路径和SHA256读取。 |
| `logs/` | 逐run日志与结果 | runtime-output | 不作为运行前置资产；当前backbone的4份选择收据已另存快照。 |

`pre_trained_model/`和`pre_trained_model_raw/`保留为新预训练的工作输出约定，不再兼任精选checkpoint发布目录。

新增或替换共享checkpoint时，必须同步更新`manifest.tsv`中的字节数和SHA256；正式入口还应自行做严格运行时校验。
