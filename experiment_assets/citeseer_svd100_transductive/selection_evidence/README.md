# SVD100 backbone选择收据

本目录是`logs/citeseer_svd100_to_cora_svd100_graphcl_135/`中4份选择收据的逐字节快照；原日志保留原位，不把135份逐run日志复制进发布资产。

| 文件 | 内容 | SHA256 |
|---|---|---|
| `citeseer_svd100_cache_receipt.json` | Citeseer source-SVD100来源与特征哈希 | `19e9c2967e71a3a4fa52fb2c87bd52bef6afe339b16b04af294c176eb57d57da` |
| `manifest.tsv` | 实验设置、source/target预处理与缓存来源收据 | `f05bc13ebab511c81e8dbd4a8a4f1c00880310acf39897d956d5cba67194e81f` |
| `per_seed_results_incremental.csv` | 135/135个checkpoint的clean validation/test结果 | `e4212ec7ec81e76f606ef60aff3360f3116ebdf83295d3b5e37f06bfaa330f49` |
| `group_summary_incremental.csv` | 严格complete-5-seed组汇总 | `91397860ada2443ae4b1ae124ba2358ac8eabf3f74e33d0d2835a5e8a76f9044` |

当前Peak由单checkpoint validation最高值选出；Stable先按完整5-seed validation mean选组，再在组内按validation选择checkpoint。test不参与这两个选择步骤。
