# 多转速、多开放度开放集实验配置说明

## 📋 概述

本文档说明了为 **CWRU** 和 **BDAE** 数据集生成的 **24 个开放集实验配置文件**。

每个配置文件对应：
- **特定转速/域**: CWRU 4个转速，BDAE 4个域
- **特定开放度**: Easy (25%), Medium (50%), Hard (75%)
- **相同训练和测试域**: 符合开放集识别的标准设置

---

## 🎯 核心设计理念

### ✅ 正确的开放集设置

- **训练集**: 仅包含已知类样本
- **测试集**: 包含已知类 + 未知类样本
- **关键**: 训练域和测试域**相同**（同一转速/工况）

### ❌ 之前的错误理解

之前错误地将"跨转速/跨域"理解为未知类的来源，但这实际上是**域适应**问题，不是开放集识别。

### ✅ 当前正确设计

- **CWRU**: 在每个转速（1797/1772/1750/1730 RPM）下分别进行开放集实验
- **BDAE**: 在每个域（E/F/G/H）下分别进行开放集实验
- **未知类**: 来自**相同工况**下的未见过的故障类型/严重程度

---

## 📊 配置文件清单

### CWRU 数据集 (12个配置)

**4个转速 × 3种开放度 = 12个实验配置**

| 转速 (RPM) | 负载 | Easy (25%) | Medium (50%) | Hard (75%) |
|------------|------|------------|--------------|------------|
| 1797 | 0 HP | `cwru_easy_rpm1797.yaml` | `cwru_medium_rpm1797.yaml` | `cwru_hard_rpm1797.yaml` |
| 1772 | 1 HP | `cwru_easy_rpm1772.yaml` | `cwru_medium_rpm1772.yaml` | `cwru_hard_rpm1772.yaml` |
| 1750 | 2 HP | `cwru_easy_rpm1750.yaml` | `cwru_medium_rpm1750.yaml` | `cwru_hard_rpm1750.yaml` |
| 1730 | 3 HP | `cwru_easy_rpm1730.yaml` | `cwru_medium_rpm1730.yaml` | `cwru_hard_rpm1730.yaml` |

### BDAE 数据集 (12个配置)

**4个域 × 3种开放度 = 12个实验配置**

| 域 | 转速范围 (LP) | Easy (25%) | Medium (50%) | Hard (75%) |
|----|--------------|------------|--------------|------------|
| E | 1000-1500 RPM | `bdae_easy_domainE.yaml` | `bdae_medium_domainE.yaml` | `bdae_hard_domainE.yaml` |
| F | 2000-3000 RPM | `bdae_easy_domainF.yaml` | `bdae_medium_domainF.yaml` | `bdae_hard_domainF.yaml` |
| G | 3500-4000 RPM | `bdae_easy_domainG.yaml` | `bdae_medium_domainG.yaml` | `bdae_hard_domainG.yaml` |
| H | 4100-5000 RPM | `bdae_easy_domainH.yaml` | `bdae_medium_domainH.yaml` | `bdae_hard_domainH.yaml` |

---

## 🔧 配置详情

## 🧩 模型与方法选择

你可以在任意 OSR 配置上切换模型和方法：

| 项 | 选项 | 说明 |
|---|---|---|
| 模型 | `lpvg_mlp` | 基于统计特征的 MLP |
| 模型 | `lpvg_gnn` | LPVG 图 + GNN |
| 模型 | `lpvg_ep_dygat` | LPVG 图 + DyGAT + Evidential |
| 方法 | `openmax` | OpenMax |
| 方法 | `crosr` | CROSR |
| 方法 | `arpl` | ARPL |
| 方法 | `evidential` | Evidential OSR |

示例：

```bash
python scripts/train.py dataset=osr_configs/cwru_easy_rpm1797 model=lpvg_ep_dygat method=evidential
```

### CWRU 配置示例

#### 1. Easy模式 @ 1797 RPM

```yaml
# cwru_easy_rpm1797.yaml
dataset:
  name: cwru_osr_easy_rpm1797
  target_rpm: 1797
  motor_load: 0 HP
  
  known_classes: [0, 3, 4, 5, 6, 7, 8, 9, 10]  # 9个已知类
  unknown_classes: [1, 2, 11]                   # 3个未知类
  openness: 0.25
  
  class_labels:
    0: Normal
    3: Inner_021
    4: Outer_007
    # ... 其他已知类
    1: Inner_007     # 未知
    2: Inner_014     # 未知
    11: Outer_028    # 未知
```

**特点**:
- 在 1797 RPM 工况下进行实验
- 已知类较多（9个），涵盖大部分故障类型
- 未知类较少（3个），随机选择的未见过的故障

#### 2. Medium模式 @ 1772 RPM

```yaml
# cwru_medium_rpm1772.yaml
dataset:
  target_rpm: 1772
  motor_load: 1 HP
  
  known_classes: 6个（随机选择）
  unknown_classes: 6个（随机选择）
  openness: 0.50
```

#### 3. Hard模式 @ 1730 RPM

```yaml
# cwru_hard_rpm1730.yaml
dataset:
  target_rpm: 1730
  motor_load: 3 HP
  
  known_classes: 4个（仅少量已知）
  unknown_classes: 8个（大量未知）
  openness: 0.75
```

---

### BDAE 配置示例

#### 1. Easy模式 @ 域E

```yaml
# bdae_easy_domainE.yaml
dataset:
  name: bdae_osr_easy_domainE
  target_domain: E
  domain_info:
    rpm_range: LP=1000-1500 RPM
    conditions: [1, 2, 3, 4, 5, 6]
    representative_rpm: 1200
  
  known_classes: [0, 1, 2]  # 3个已知故障类型
  unknown_classes: [3]       # 1个未知故障类型
  openness: 0.25
  
  class_labels:
    0: Normal_DomainE
    1: Inner_0.5-0.5_DomainE
    2: Inner_0.5-1.0_DomainE
    3: Outer_0.5-0.5_DomainE  # 未知
```

**特点**:
- 在域E（低速工况）下进行实验
- 已知3个故障类型，未知1个故障类型
- 所有数据都来自同一转速域（LP=1000-1500 RPM）

#### 2. Medium模式 @ 域F

```yaml
# bdae_medium_domainF.yaml
dataset:
  target_domain: F
  domain_info:
    rpm_range: LP=2000-3000 RPM
    conditions: [7, 8, 9, 10, 11, 12, 13, 14]
  
  known_classes: [0, 1]  # 2个已知
  unknown_classes: [2, 3]  # 2个未知
  openness: 0.50
```

#### 3. Hard模式 @ 域H

```yaml
# bdae_hard_domainH.yaml
dataset:
  target_domain: H
  domain_info:
    rpm_range: LP=4100-5000 RPM
  
  known_classes: [0]  # 仅1个已知（通常是Normal）
  unknown_classes: [1, 2, 3]  # 3个未知
  openness: 0.75
```

---

## 📈 实验设计

### 开放度分层

| 开放度 | 已知类比例 | 未知类比例 | 适用场景 |
|--------|-----------|-----------|---------|
| **Easy (25%)** | 75% | 25% | 初步验证、方法对比 |
| **Medium (50%)** | 50% | 50% | 标准实验、论文发表 |
| **Hard (75%)** | 25% | 75% | 鲁棒性测试、真实场景 |

### 随机类别分配

- **已知类**: 随机选择（但始终包含 Normal 类）
- **未知类**: 从剩余类别中随机选择
- **随机种子**: 设置为 42，确保可复现

### 数据划分

所有配置统一使用：
```yaml
train_ratio: 0.6   # 60% 训练（仅已知类）
val_ratio: 0.2     # 20% 验证
test_ratio: 0.2    # 20% 测试（已知+未知类）
unknown_ratio: 开放度值  # 测试集中未知类的比例
```

---

## 🚀 使用方法

### 1. 单个实验（Hydra 入口）

```bash
# CWRU @ 1797 RPM, Easy模式
python scripts/train.py dataset=osr_configs/cwru_easy_rpm1797

# BDAE @ 域F, Medium模式
python scripts/train.py dataset=osr_configs/bdae_medium_domainF
```

### 2. 批量实验

#### CWRU 所有转速，单一开放度
```bash
for rpm in 1797 1772 1750 1730; do
  python scripts/train.py dataset=osr_configs/cwru_medium_rpm${rpm}
done
```

#### CWRU 单一转速，所有开放度
```bash
for level in easy medium hard; do
  python scripts/train.py dataset=osr_configs/cwru_${level}_rpm1797
done
```

#### 运行所有 CWRU 实验 (12个)
```bash
for rpm in 1797 1772 1750 1730; do
  for level in easy medium hard; do
    python scripts/train.py dataset=osr_configs/cwru_${level}_rpm${rpm}
  done
done
```

#### 运行所有 BDAE 实验 (12个)
```bash
for domain in E F G H; do
  for level in easy medium hard; do
    python scripts/train.py dataset=osr_configs/bdae_${level}_domain${domain}
  done
done
```

### 3. Python 脚本调用

```python
import yaml

config_path = "LPVG_OSR/config/dataset/osr_configs/cwru_easy_rpm1797.yaml"
with open(config_path, 'r') as f:
  config = yaml.safe_load(f)

dataset_name = config['dataset']['name']
known_classes = config['dataset']['known_classes']
unknown_classes = config['dataset']['unknown_classes']
target_rpm = config['dataset']['target_rpm']

print(f"数据集: {dataset_name}")
print(f"目标转速: {target_rpm} RPM")
print(f"已知类: {known_classes}")
print(f"未知类: {unknown_classes}")
```

---

## 📊 实验建议

### 阶段1: 快速验证（选择性实验）

选择代表性配置进行初步验证：

```bash
# CWRU: 1797 RPM (0 HP) 所有开放度
python run_experiments.py --config LPVG_OSR/config/dataset/osr_configs/cwru_easy_rpm1797.yaml
python run_experiments.py --config LPVG_OSR/config/dataset/osr_configs/cwru_medium_rpm1797.yaml
python run_experiments.py --config LPVG_OSR/config/dataset/osr_configs/cwru_hard_rpm1797.yaml

# BDAE: 域E (低速) 所有开放度
python run_experiments.py --config LPVG_OSR/config/dataset/osr_configs/bdae_easy_domainE.yaml
python run_experiments.py --config LPVG_OSR/config/dataset/osr_configs/bdae_medium_domainE.yaml
python run_experiments.py --config LPVG_OSR/config/dataset/osr_configs/bdae_hard_domainE.yaml
```

### 阶段2: 转速/域影响分析

固定开放度，比较不同转速/域的影响：

```bash
# CWRU Medium 模式下的转速影响
for rpm in 1797 1772 1750 1730; do
    python run_experiments.py --config LPVG_OSR/config/dataset/osr_configs/cwru_medium_rpm${rpm}.yaml
done

# BDAE Medium 模式下的域影响
for domain in E F G H; do
    python run_experiments.py --config LPVG_OSR/config/dataset/osr_configs/bdae_medium_domain${domain}.yaml
done
```

### 阶段3: 完整实验（论文级别）

运行所有 24 个配置，获取完整的实验数据：

```bash
# 运行所有实验
bash LPVG_OSR/scripts/run_all_osr_experiments.sh
```

---

## 📈 期望性能指标

### CWRU 数据集

| 转速 (RPM) | 开放度 | Known Acc | Unknown Recall | H-score | AUROC |
|-----------|--------|-----------|----------------|---------|-------|
| 1797 | Easy | > 93% | > 88% | > 90% | > 0.92 |
| 1797 | Medium | > 90% | > 85% | > 87% | > 0.90 |
| 1797 | Hard | > 85% | > 80% | > 82% | > 0.85 |
| 1772 | Easy | > 92% | > 87% | > 89% | > 0.91 |
| ... | ... | ... | ... | ... | ... |

### BDAE 数据集

| 域 | 开放度 | Known Acc | Unknown Recall | H-score | AUROC |
|----|--------|-----------|----------------|---------|-------|
| E | Easy | > 91% | > 86% | > 88% | > 0.90 |
| E | Medium | > 88% | > 82% | > 85% | > 0.87 |
| E | Hard | > 83% | > 76% | > 79% | > 0.83 |
| F | Easy | > 90% | > 85% | > 87% | > 0.89 |
| ... | ... | ... | ... | ... | ... |

**注**: 不同转速/域可能有性能差异，高速域通常更具挑战性。

---

## 🔍 配置文件结构

每个配置文件都包含以下标准字段：

```yaml
dataset:
  name: <experiment_name>
  data_root: <path_to_data>
  
  # 转速/域标识
  target_rpm: <rpm>  # CWRU
  target_domain: <domain>  # BDAE
  
  # 开放集配置
  known_classes: [...]
  unknown_classes: [...]
  openness: <value>
  
  # 类别标签
  class_labels:
    <id>: <label>
  
  # 数据划分
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
  unknown_ratio: <openness>
  
  # 信号参数
  sampling_rate: <value>
  signal_length: 2048
  overlap: 0.5
  
  # 预处理
  preprocessing:
    normalization: zscore
    method: fft
  
  # 备注
  notes: <description>
```

---

## 📁 文件位置

所有配置文件保存在：
```
LPVG_OSR/config/dataset/osr_configs/
├── cwru_easy_rpm1797.yaml
├── cwru_easy_rpm1772.yaml
├── cwru_easy_rpm1750.yaml
├── cwru_easy_rpm1730.yaml
├── cwru_medium_rpm1797.yaml
├── ... (共12个CWRU配置)
├── bdae_easy_domainE.yaml
├── bdae_easy_domainF.yaml
├── bdae_easy_domainG.yaml
├── bdae_easy_domainH.yaml
├── bdae_medium_domainE.yaml
├── ... (共12个BDAE配置)
```

---

## ✅ 验证清单

- [x] 24 个配置文件全部生成
- [x] 训练域和测试域相同（符合开放集定义）
- [x] 未知类来自同一工况的未见过故障
- [x] 已知类随机分配（包含 Normal）
- [x] 开放度正确计算（25%, 50%, 75%）
- [x] 数据划分比例合理（60/20/20）
- [x] unknown_ratio 与开放度一致
- [x] 预处理参数统一（FFT + zscore）

---

## 🎓 重要说明

### 开放集 vs 域适应

- **开放集识别**: 训练域 = 测试域，未知类来自**未见过的类别**
- **域适应**: 训练域 ≠ 测试域，未知类来自**不同的工况**

本项目是**开放集识别**，因此所有配置都是在**同一转速/域**下进行的。

### 跨转速/跨域实验

如果需要研究**跨转速泛化**或**域适应**，需要单独设计配置文件，例如：
- 训练域: 1797 RPM
- 测试域: 1772 RPM
- 这属于**域适应 + 开放集**的组合问题，更具挑战性

---

## 📚 相关文档

1. **配置生成脚本**: `LPVG_OSR/scripts/generate_osr_configs.py`
2. **批量实验脚本**: `LPVG_OSR/scripts/run_all_osr_experiments.sh`
3. **开放集任务设置**: `docs/OPEN_SET_TASK_SETUP.md`

---

**最后更新**: 2026年2月11日
**版本**: v2.0 (修正版)
**状态**: ✅ 完成 - 24个配置，符合开放集识别标准定义
