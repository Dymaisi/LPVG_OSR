# OSR 实验配置总表（CWRU + BDAE）

命令格式：

```
python scripts/train.py dataset=osr_configs/<config_name> model=<lpvg_mlp|lpvg_gnn> method=<openmax|crosr|arpl>
```

## CWRU（12 个配置）

| 配置 | 转速/负载 | Openness | 已知类 | 未知类 | 命令 |
|---|---|---:|---|---|---|
| cwru_easy_rpm1797 | 1797 / 0 HP | 0.25 | [0,3,4,5,6,7,8,9,10] | [1,2,11] | `python scripts/train.py dataset=osr_configs/cwru_easy_rpm1797 model=lpvg_mlp method=openmax` |
| cwru_easy_rpm1772 | 1772 / 1 HP | 0.25 | [0,1,2,4,7,8,9,10,11] | [3,5,6] | `python scripts/train.py dataset=osr_configs/cwru_easy_rpm1772 model=lpvg_mlp method=openmax` |
| cwru_easy_rpm1750 | 1750 / 2 HP | 0.25 | [0,1,3,5,6,8,9,10,11] | [2,4,7] | `python scripts/train.py dataset=osr_configs/cwru_easy_rpm1750 model=lpvg_mlp method=openmax` |
| cwru_easy_rpm1730 | 1730 / 3 HP | 0.25 | [0,1,3,6,7,8,9,10,11] | [2,4,5] | `python scripts/train.py dataset=osr_configs/cwru_easy_rpm1730 model=lpvg_mlp method=openmax` |
| cwru_medium_rpm1797 | 1797 / 0 HP | 0.50 | [0,3,4,5,6,11] | [1,2,7,8,9,10] | `python scripts/train.py dataset=osr_configs/cwru_medium_rpm1797 model=lpvg_mlp method=openmax` |
| cwru_medium_rpm1772 | 1772 / 1 HP | 0.50 | [0,2,3,7,8,9] | [1,4,5,6,10,11] | `python scripts/train.py dataset=osr_configs/cwru_medium_rpm1772 model=lpvg_mlp method=openmax` |
| cwru_medium_rpm1750 | 1750 / 2 HP | 0.50 | [0,1,7,8,9,11] | [2,3,4,5,6,10] | `python scripts/train.py dataset=osr_configs/cwru_medium_rpm1750 model=lpvg_mlp method=openmax` |
| cwru_medium_rpm1730 | 1730 / 3 HP | 0.50 | [0,1,2,7,8,9] | [3,4,5,6,10,11] | `python scripts/train.py dataset=osr_configs/cwru_medium_rpm1730 model=lpvg_mlp method=openmax` |
| cwru_hard_rpm1797 | 1797 / 0 HP | 0.75 | [0,2,6,10] | [1,3,4,5,7,8,9,11] | `python scripts/train.py dataset=osr_configs/cwru_hard_rpm1797 model=lpvg_mlp method=openmax` |
| cwru_hard_rpm1772 | 1772 / 1 HP | 0.75 | [0,3,8,10] | [1,2,4,5,6,7,9,11] | `python scripts/train.py dataset=osr_configs/cwru_hard_rpm1772 model=lpvg_mlp method=openmax` |
| cwru_hard_rpm1750 | 1750 / 2 HP | 0.75 | [0,3,10,11] | [1,2,4,5,6,7,8,9] | `python scripts/train.py dataset=osr_configs/cwru_hard_rpm1750 model=lpvg_mlp method=openmax` |
| cwru_hard_rpm1730 | 1730 / 3 HP | 0.75 | [0,3,9,11] | [1,2,4,5,6,7,8,10] | `python scripts/train.py dataset=osr_configs/cwru_hard_rpm1730 model=lpvg_mlp method=openmax` |

## BDAE（12 个配置）

| 配置 | 域 | Openness | 已知类 | 未知类 | 命令 |
|---|---|---:|---|---|---|
| bdae_easy_domainE | E | 0.25 | [0,2,3] | [4] | `python scripts/train.py dataset=osr_configs/bdae_easy_domainE model=lpvg_mlp method=openmax` |
| bdae_easy_domainF | F | 0.25 | [0,2,1] | [4] | `python scripts/train.py dataset=osr_configs/bdae_easy_domainF model=lpvg_mlp method=openmax` |
| bdae_easy_domainG | G | 0.25 | [0,1,2] | [4] | `python scripts/train.py dataset=osr_configs/bdae_easy_domainG model=lpvg_mlp method=openmax` |
| bdae_easy_domainH | H | 0.25 | [0,2,1] | [4] | `python scripts/train.py dataset=osr_configs/bdae_easy_domainH model=lpvg_mlp method=openmax` |
| bdae_medium_domainE | E | 0.50 | [0,1] | [4,5] | `python scripts/train.py dataset=osr_configs/bdae_medium_domainE model=lpvg_mlp method=openmax` |
| bdae_medium_domainF | F | 0.50 | [0,1] | [4,5] | `python scripts/train.py dataset=osr_configs/bdae_medium_domainF model=lpvg_mlp method=openmax` |
| bdae_medium_domainG | G | 0.50 | [0,1] | [4,5] | `python scripts/train.py dataset=osr_configs/bdae_medium_domainG model=lpvg_mlp method=openmax` |
| bdae_medium_domainH | H | 0.50 | [0,2] | [4,5] | `python scripts/train.py dataset=osr_configs/bdae_medium_domainH model=lpvg_mlp method=openmax` |
| bdae_hard_domainE | E | 0.75 | [0] | [4,5,6] | `python scripts/train.py dataset=osr_configs/bdae_hard_domainE model=lpvg_mlp method=openmax` |
| bdae_hard_domainF | F | 0.75 | [0] | [4,5,6] | `python scripts/train.py dataset=osr_configs/bdae_hard_domainF model=lpvg_mlp method=openmax` |
| bdae_hard_domainG | G | 0.75 | [0] | [4,5,6] | `python scripts/train.py dataset=osr_configs/bdae_hard_domainG model=lpvg_mlp method=openmax` |
| bdae_hard_domainH | H | 0.75 | [0] | [4,5,6] | `python scripts/train.py dataset=osr_configs/bdae_hard_domainH model=lpvg_mlp method=openmax` |

## BDAE 域设置说明（E/F/G/H）

| 域 | 转速范围 (LP) | 工况条件 | 代表转速 | 说明 |
|---|---|---|---|---|
| E | 1000-1500 RPM | 1-6 | 1200 | 低速域 |
| F | 2000-3000 RPM | 7-14 | 2500 | 中速域 |
| G | 3500-4000 RPM | 15-20 | 3750 | 高速域 |
| H | 4100-5000 RPM | 21-27 | 4500 | 超高速域 |

说明：
1. 域 E/F/G/H 不仅是转速范围，还包含对应的工况条件集合。
2. “工况条件”数字是 BDAE 数据集内定义的工况编号（working condition IDs，范围 1-27）。
3. 代表转速用于概括该域的典型运行点。

## 模型与方法可选项

| 类型 | 可选值 | 说明 |
|---|---|---|
| model | lpvg_mlp | 基于图特征的 MLP 分类器 |
| model | lpvg_gnn | 基于图结构的 GNN 分类器 |
| model | lpvg_ep_dygat | LPVG-EP-DyGAT 端到端模型（DyGAT + 证据原型） |
| method | openmax | OpenMax 开放集方法 |
| method | crosr | CROSR 开放集方法 |
| method | arpl | ARPL 开放集方法 |
| method | evidential | 证据学习开放集方法（配合 lpvg_ep_dygat） |

用法示例：

```
python scripts/train.py dataset=osr_configs/cwru_easy_rpm1797 model=lpvg_gnn method=crosr
python scripts/train.py dataset=osr_configs/cwru_easy_rpm1797 model=lpvg_ep_dygat method=evidential
```
