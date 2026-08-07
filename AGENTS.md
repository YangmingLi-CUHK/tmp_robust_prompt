# AGENTS.md

> Codex在本仓库的长期工作记忆。最后更新：2026-08-04。这里只保留会影响后续判断的稳定事实、当前主线和操作纪律；历史细节查报告，不在此复制。

## 1.目标与当前问题

项目目标是在`GNN自监督预训练+Prompt Learning`范式下，提高节点分类面对Metattack结构污染时的鲁棒性。

当前主线不是继续盲调filter或loss，而是先回答三个基础问题：

1. 预训练与下游prompt是否必须来自同一数据集；
2. prompt实际作用于原始输入特征还是GCL embedding；
3. Cora与Citeseer原始维数不同时，跨数据集迁移究竟只是shape兼容，还是具有可用的表示迁移。

当前只讨论Cora/Citeseer。除非用户重新指定，不扩展到化学/生物图。

## 2.不可违背的工作纪律

- **`CLAUDE.md`只读**：可参考，但未经用户明确要求，不得修改、格式化、重编码或touch该文件。
- **证据分层**：区分论文声称、官方代码行为、本仓代码行为、真实日志/结果和我们的推断；不能互相替代。
- **运行事实优先**：旧新脚本并存，部分服务器实验没有`.sh`。判断“实际跑了什么”时，联合检查mtime、log、结果表、checkpoint、调用链和git版本，不能只看文件名。
- **尊重数字，质疑setting**：真实日志中的数字视为真实运行结果；可以质疑数据源、split、preprocess、泄漏、评估口径和解释，但不能随意否定数字。
- **逐层核对维数**：至少写清`data.x→GCN第一层→hidden/embedding→prompt→head`，同时核对checkpoint首层shape。
- **最小改动**：不重构无关代码，不引入新依赖，不覆盖用户改动；修改前先列计划，修改后做与风险相称的检查。
- **重计算放服务器**：本地以只读审计、静态检查和小型smoke test为主。GitHub同步成本高，上传前应把参数、路径、缓存和续跑逻辑查清。
- **报告口径**：accuracy使用全部5个seed的简单mean±population std，不删除最小值；按validation mean选配置，test只用于最终报告，禁止test-oracle选型。
- **报告风格**：克制、技术优先、少remark；中文与英文字母/数字/符号之间不额外插空格。每次结果必须注明clean/ptb、数据路径、split、preprocess、backbone、checkpoint和完整维数链。
- 污染浓度统一写作`0.00/0.05/0.10/0.15/0.20/0.25`；遇到`0/0.0/0.00`或`0.2/0.20`并存时，先沿最新调用链和日志确认，不能猜。

## 3.已经确认的阶段性认知

### 3.1 AttributePrompt

其高鲁棒性主要来自clean图上训练的两层GCN，以及随后被冻结的既有embedding/分类头；污染图没有有效改变这条固定预测链。该结果说明“固定clean表示”可以表面稳定，但作为可适应污染图的prompt防御对比，含义有限。

### 3.2 GPromptShield/RobustPrompt-T

- 论文与代码均支持：GNN在clean图上预训练，downstream冻结backbone。
- RobustPrompt-T把prompt加在**GCN之前的输入特征空间**。以Cora为例是1433维，不是GraphCL输出的256维embedding。
- 在官方论文/代码中尚未找到citation数据集跨域时处理`D_source≠D_target`的明确转换；现有设定实质上以同数据集pretrain/downstream为主。
- 原版与本仓增强版并不等价：原版`τ_tune`使用prompt后特征cosine且单次forward、attention输出被`F.normalize`覆写、`out_detect_pt`为`pass`、MSE基于剪枝前全图、常用`hid_dim=64`；本仓增强版曾改为embedding cosine/两次forward、真attention、实现OOD、剪枝后MSE和`hid_dim=256`。比较结果前必须明确`--prompt_variant`。
- IA-PT会把`idx_test`节点生成的伪标签并入训练，存在评估split污染风险；这不是“使用真测试标签”，也不能未经隔离实验就把论文精度完全归因于泄漏。
- Meeting15显示的主要风险包括：少标签下大token bank过拟合、稠密prompt扰动稀疏BoW、KL目标可能抵消防御、粗粒度prompt组合退化。它们是代码/实验支持的风险解释，不应写成已完成因果证明。
- `sim_pt`是目前较有信息量的攻击边检测器，但“检测边较好”没有自动转化为“分类鲁棒”；多filter组合通常损伤clean性能。
- Metattack预算曾因self-loop计数膨胀约25.7%，后续已按无自环边数修正并重生成数据。仓库仍有旧文件/旧脚本，不能把这项修复泛化到所有历史结果。

### 3.3 相关prompt范式

- GPPT官方实现按数据集分别训练，原始输入维数保持为各自的`D`，两层GCN为`D→128→128`；两个128是连续层宽，不是两个embedding拼成256。它没有提供Cora/Citeseer跨域input统一范式。
- All-in-One在原始节点特征上做SVD100，再进入encoder。这提供了shape统一的先例，但会直接改变原始特征空间，不能视为天然的跨域语义对齐。

## 4.当前数据与维数真值

- **Cora当前下游**：full clean Cora，`2708×1433`，7类；5-shot/split-1为`35/265/2408`。Cora-LCC问题暂缓，不要悄悄切换到2485节点版本。
- **Citeseer当前预训练源**：DeepRobust/Nettack LCC，`2110×3703`，6类，3668条无向边。
- 现有预训练预处理顺序是`raw BoW→L1行归一化→可选SVD→METIS 200→GraphCL`，不是未归一化raw BoW直接SVD。
- PyG的`SVDFeatureReduction`在当前实现中使用exact`torch.linalg.svd`。
- Citeseer与Cora分别拟合SVD时，只统一shape；两者不共享`V`或坐标基底，不能声称完成语义对齐。
- 当前GraphCL的两层encoder为`input_dim→256→256`；projection head为`256→256→256`，只参与对比预训练且不写入checkpoint。下游logistic head接收256维embedding。
- frozen linear probe仍先用GCN和图结构生成embedding，不能再描述成“不使用图结构”。

## 5.已经完成的跨数据集实验

### Citeseer-SVD1433→Cora-clean

流程：`Citeseer 3703→L1→SVD1433→GraphCL(1433→256→256)`；Cora保持`L1-BoW 1433维`、不做SVD，然后经过冻结encoder和`LogisticRegression(256→7)`。

135/135个checkpoint均完成clean评估。按5-seed validation mean选出的`dropN/dropN,ratio=0.1`在test上为`0.2977±0.0370`，低于test-majority baseline`0.3040`。该实验没有污染图结果。结论只到：**shape兼容不足以带来可用迁移**；不能据此归因于Cora-LCC，也不能称为鲁棒性结论。

简报：`neo_report/citeseer_cora_graphcl_clean_brief_20260730.html`。

## 6.已完成clean选型：Citeseer-SVD100→Cora-SVD100

实验假设：把两端都压到100维，测试按奇异分量次序形成的低维表示能否比“Citeseer-SVD1433→Cora原生1433”更适合迁移。它仍是**independent-SVD、shape-only**实验。

```text
Citeseer raw3703→L1→独立SVD100→GraphCL GCN(100→256→256)
→保存encoder
Cora raw1433→L1→另行独立SVD100→冻结同一encoder
→256维node embedding→全节点z-score→multinomial LogisticRegression
```

网格是135次run，不是135个独立超参组：

- `aug1,aug2∈{dropN,permE,maskN}`，有序且允许相同；
- `ratio∈{0.1,0.2,0.3}`；
- pretrain seed为1–5；
- 固定`lr=0.001,epochs=200,GCN layers=2,hid=256,batch=10,T=0.1,weight_decay=0.0001`；
- 共27组增强配置，每组严格5个pretrain seed。

入口：

```bash
bash run_citeseer_svd100_to_cora_svd100_graphcl_135.sh
```

原始运行与发布收据：

- `run_citeseer_svd100_to_cora_svd100_graphcl_135.sh`
- `eval_citeseer_svd100_to_cora_svd100.py`
- `experiment_assets/citeseer_svd100_transductive/selection_evidence/manifest.tsv`
- `experiment_assets/citeseer_svd100_transductive/selection_evidence/per_seed_results_incremental.csv`
- `experiment_assets/citeseer_svd100_transductive/selection_evidence/group_summary_incremental.csv`
- `experiment_assets/citeseer_svd100_transductive/selection_evidence/citeseer_svd100_cache_receipt.json`

脚本会校验source/target SVD缓存来源、checkpoint结构与SHA256，增量原子写结果，并在续跑时重建summary。没有成功CSV收据的同名checkpoint视为来源不明，会保留改名后重训。

**真实结果**：135/135个checkpoint均成功完成clean评估。

- Peak BB：全部135个单checkpoint中validation最高，`dropN/dropN,ratio=0.1,seed=1`；validation=`0.535849`，test=`0.531561`。
- Stable BB：先按完整5-seed validation mean选出第一组`maskN/dropN,ratio=0.2`，组validation为`0.464151±0.044522`（population std），再在组内按validation选出seed1；该checkpoint validation=`0.516981`，test=`0.559801`。
- 两个选择步骤都不使用test。两个独立SVD仍只对齐shape，不共享语义基底。
- 两份发布权重、字节数、SHA256、维数和历史路径统一登记在`experiment_assets/manifest.tsv`。

## 7.当前待运行主线：2BB×3方法×6浓度Transductive

当前实验固定使用上述Peak/Stable两个backbone，对比：

- GPromptShield经典版匹配设置；
- 本仓更改版GPromptShield；
- 冻结同一GraphCL encoder、无filter的GPPT。

污染浓度统一为`0.00/0.05/0.10/0.15/0.20/0.25`，每个`backbone×method×rate`运行5个downstream seed，因此总计`2×3×6×5=180`次。

入口：

```bash
bash run_citeseer_svd100_to_cora_svd100_transductive_2bb_3methods_corrected_budget_180.sh
```

关键事实：

- 控制器：`run_citeseer_svd100_to_cora_svd100_transductive_2bb_3methods_corrected_budget_180.py`。
- 两份当前权重位于`experiment_assets/citeseer_svd100_transductive/`，并按固定SHA256和`GCN(100→256→256)`严格校验。
- Cora target-SVD100缓存缺失时自动精确重算并原子写入`data/preprocessed/cora_clean_full_l1_svd_100.pt`；缓存是可重算产物，不进入Git资产。
- 六张corrected-budget污染图、feature、label和split均按固定SHA256预检；浓度命名不得回退为`0.0/0.1/0.2`等别名。
- 输出目录为`logs/citeseer_svd100_to_cora_svd100_transductive_2bb_3methods_corrected_budget_180/`，重复运行同一入口按成功CSV收据续跑。
- 本地已通过静态、180计划、资产manifest、选择收据复算和Git归档模拟；由于本机没有PyTorch环境，正式`torch.load(strict=True)`、clean replay与180次GPU运行仍由服务器preflight完成。

服务器顺序：先运行`PREFLIGHT_ONLY=1`，确认两份权重、SVD缓存、clean replay及污染图全部通过，再启动正式总控。最终结果按每组5个seed的test mean±population std报告，不删除seed；表格与可视化保持极简，不做超出证据的防御机制归因。

## 8.常见陷阱

- `eval_pretrain.py`不会给Cora做SVD，且旧批量逻辑按test accuracy排序；不能用于正式SVD100选型。
- `MyTask.py`虽然接收preprocess参数，现有NodeTask链路不会自动对downstream Cora执行SVD；因此当前使用专用evaluator。
- GraphCL文件名没有完整编码epochs、层数、METIS parts、batch、temperature或数据hash；不能仅凭同名文件认定setting一致。
- GraphCL保存的是最终epoch的GNN，不是最佳loss模型，也不保存projection head；增强视图在训练前生成一次并在全部epoch复用。
- 高污染下性能崩塌不等于某个filter阈值没调好；已有证据表明图传播本身可能成为主要风险源。

## 9.最小索引

- 入口：`MyPretrain.py`、`MyTask.py`
- 数据：`prompt_graph/data/load4data.py`
- GraphCL：`prompt_graph/pretrain/GraphCL.py`
- 任务：`prompt_graph/tasker/task.py`、`prompt_graph/tasker/node_task.py`
- Prompt：`prompt_graph/prompt/RobustPrompt_T.py`、`RobustPrompt_T_original.py`、`RobustPrompt_T_NSP.py`
- Meeting15审计：`advanced_report/meeting15_root_cause.html`、`advanced_report/meeting15_diagnostics.html`
- 原始维数链：`reports/data_pipeline_matrix_flow.html`
- 官方代码快照：`reference/code_sources/`
- Git协作背景：`advanced_report/git_related.md`；实际同步前仍须运行`git remote -v`与`git status`核验当前状态。

远程运行环境可能变化；服务器路径、conda环境和GPU在执行前现场核验，不把旧文档中的机器状态当永久事实。
