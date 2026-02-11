# 开放集识别方法说明

## 1. 问题定义

### 闭集识别 (Closed Set Recognition)
- 训练集和测试集包含相同的类别
- 目标：将样本分类到已知类别之一

### 开放集识别 (Open Set Recognition)
- 训练集只包含已知类别
- 测试集包含已知类别 + 未知类别
- 目标：
  1. 正确分类已知类别
  2. 拒绝（识别）未知类别

## 2. 方法概述

### 2.1 OpenMax

**原理**: 使用 Weibull 分布建模激活值分布，估计未知类的概率

**步骤**:
1. 训练闭集分类器
2. 为每个类别拟合 Weibull 分布（基于 MAV - Mean Activation Vector）
3. 测试时计算 OpenMax 分数（调整后的 softmax）
4. 低于阈值的样本拒绝为未知

**优点**:
- 无需重新训练模型
- 理论基础扎实

**缺点**:
- 需要调整多个超参数

### 2.2 CROSR (Classification-Reconstruction)

**原理**: 结合分类和重构任务，未知类难以重构

**架构**:
```
输入 → 编码器 → 特征 → 分类器 → 类别
              ↓
         解码器 → 重构
```

**损失函数**:
$$
L = L_{cls} + \lambda L_{recon}
$$

**检测未知**:
- 重构误差高 → 未知类

**优点**:
- 端到端训练
- 重构提供额外信息

**缺点**:
- 计算开销大
- 需要平衡两个损失

### 2.3 ARPL (Adversarial Reciprocal Points Learning)

**原理**: 为每个已知类学习一个"互反点"（Reciprocal Point），未知类远离所有互反点

**核心思想**:
1. 已知类样本应该接近自己类别的互反点
2. 未知类样本远离所有互反点
3. 使用对抗训练增强泛化

**损失函数**:
$$
L = L_{cls} + \lambda_{RPL} L_{RPL} + \lambda_{adv} L_{adv}
$$

**优点**:
- 显式建模未知类
- 效果优秀

**缺点**:
- 训练复杂
- 对抗训练不稳定

## 3. 评估指标

### 3.1 闭集指标

- **Accuracy**: 已知类分类准确率
- **F1-score**: 已知类 F1 分数

### 3.2 开放集指标

- **AUROC**: 已知/未知二分类的 ROC 曲线下面积
- **Open Set F1**: 将未知类作为额外类别的 F1 分数
- **OSCR**: 在固定假阳率下的分类率
- **Unknown Recall**: 未知类检测率

## 4. 方法选择建议

| 场景 | 推荐方法 | 理由 |
|-----|---------|------|
| 快速原型 | OpenMax | 简单，无需重新训练 |
| 高精度要求 | ARPL | 效果最好 |
| 计算资源有限 | OpenMax | 轻量级 |
| 有重构先验 | CROSR | 利用重构信息 |

## 5. 参考文献

1. Bendale, A., & Boult, T. (2016). "Towards open set deep networks." CVPR.
2. Yoshihashi, R., et al. (2019). "Classification-reconstruction learning for open-set recognition." CVPR.
3. Chen, G., et al. (2021). "Learning open set network with discriminative reciprocal points." ECCV.
