# All-in-One／GPPT材料来源

检索日期：2026-07-30。此目录仅保存研究材料；未改动项目实验代码、数据或权重。

## All-in-One

|材料|本地路径|来源与版本|
|---|---|---|
|论文|`../All_in_One.pdf`|arXiv:[All in One:Multi-Task Prompting for Graph Neural Networks](https://arxiv.org/abs/2307.01504)；SHA256:`C36D1188BBCEDD1898DFE05CC21F897CB0ADAEA2317F046856886A93766E9AA9`|
|匿名投稿原始代码|`All_in_One_raw_mpg/`、`All_in_One_raw_mpg.zip`|[Anonymous GitHub](https://anonymous.4open.science/r/mpg)；页面revision:`de0ccb85`；15个文件；ZIP SHA256:`7D9F89BCD4B884190226B0DAC13F274C8AE5CBFDE9A6E7B9A895E3A801B38F36`|
|作者后续代码|`All_in_One_ProG_ori_code/`|[sheldonresearch/ProG](https://github.com/sheldonresearch/ProG)的`ori`分支；commit:`0fa942226f5e0a2b0f0c9271b59d563c2e9c7c38`|

注意：匿名原始包最接近论文投稿时实现；`ProG/ori`的README明确称其为重新设计、改动很大的后续版本，不能把两者视为同一快照。

## GPPT

|材料|本地路径|来源与版本|
|---|---|---|
|论文入口|—|ACM:[GPPT:Graph Pre-training and Prompt Tuning to Generalize Graph Neural Networks](https://doi.org/10.1145/3534678.3539249)|
|作者方法文本|`../GPPT_author_dissertation_Kaixiong_Zhou_2023.pdf`|[Rice University博士论文](https://repository.rice.edu/bitstream/handle/1911/115249/ZHOU-DOCUMENT-2023.pdf?sequence=1)；Chapter7为GPPT完整方法章；SHA256:`3668EC5324DF1C182191FDCC4E4C8EB795B1FE5B5617D0894033E36690533010`|
|方法章摘录|`../GPPT_KDD2022_author_dissertation_chapter7.pdf`|从上述论文PDF第112–133页抽取，共22页；SHA256:`BCC786B1CD6758DE1C1D42EE7E8C2CE4E10F351DAC5F6DF74DBA4B1DAF581672`|
|作者代码|`GPPT_official/`、`GPPT_official_main.zip`|[MingChen-Sun/GPPT](https://github.com/MingChen-Sun/GPPT)；commit:`4f919ab66c5c2edebcc0356cf6399074c24a99a4`；ZIP SHA256:`65A7CBF6A10039CB2558925F8E6E47D928A0D9EC9D0283EC041A8DAFCFBB8126`|

ACM独立PDF下载端点在本次检索中返回403；因此方法逐式核验采用共同作者Kaixiong Zhou博士论文的GPPT完整章节，并保留ACM DOI作为正式出版入口。
