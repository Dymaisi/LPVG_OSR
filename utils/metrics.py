"""
开放集识别评估指标
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, Tuple, Any


def compute_osr_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    is_unknown: np.ndarray
) -> Dict[str, float]:
    """计算开放集识别指标
    
    Args:
        labels: 真实标签
        predictions: 预测标签
        scores: 预测分数 (N, num_classes)
        is_unknown: 是否为未知类
        
    Returns:
        metrics: 指标字典
    """
    metrics = {}
    
    # 已知类 mask
    known_mask = ~is_unknown.astype(bool)
    
    # 1. 闭集准确率（只在已知类上计算）
    if known_mask.sum() > 0:
        metrics['accuracy'] = accuracy_score(
            labels[known_mask],
            predictions[known_mask]
        )
        metrics['known_acc'] = metrics['accuracy']  # 别名
    else:
        metrics['accuracy'] = 0.0
        metrics['known_acc'] = 0.0
    
    # 2. F1 分数（已知类）
    if known_mask.sum() > 0:
        metrics['f1_known'] = f1_score(
            labels[known_mask],
            predictions[known_mask],
            average='macro',
            zero_division=0
        )
    else:
        metrics['f1_known'] = 0.0
    
    # 3. AUROC（已知 vs 未知）
    # 使用最大分数作为置信度
    max_scores = scores.max(axis=1)
    
    # 创建二分类标签：0=未知，1=已知
    binary_labels = (~is_unknown).astype(int)
    
    try:
        metrics['auroc'] = roc_auc_score(binary_labels, max_scores)
    except:
        metrics['auroc'] = 0.0
    
    # 4. 开放集 F1（将未知类作为一个额外类别）
    # 为未知类分配一个新的标签
    modified_labels = labels.copy()
    modified_preds = predictions.copy()
    
    max_known_label = labels[known_mask].max() if known_mask.sum() > 0 else 0
    unknown_label = max_known_label + 1
    
    # 标记真实未知类
    modified_labels[is_unknown] = unknown_label
    
    # 使用阈值判断预测是否为未知
    # 如果最大分数低于阈值，认为是未知类
    threshold = np.percentile(max_scores[known_mask], 95) if known_mask.sum() > 0 else 0.5
    predicted_unknown = max_scores < threshold
    modified_preds[predicted_unknown] = unknown_label
    
    try:
        metrics['f1_openset'] = f1_score(
            modified_labels,
            modified_preds,
            average='macro',
            zero_division=0
        )
    except:
        metrics['f1_openset'] = 0.0
    
    # 5. OSCR (Open Set Classification Rate)
    # 在不同的假阳率下，正确分类已知类的比率
    metrics['oscr'] = compute_oscr(
        labels,
        predictions,
        max_scores,
        is_unknown
    )
    
    # 6. 未知类检测率（召回率）
    if is_unknown.sum() > 0:
        unknown_detected = predicted_unknown[is_unknown].sum()
        metrics['unknown_recall'] = unknown_detected / is_unknown.sum()
        metrics['unknown_acc'] = metrics['unknown_recall']  # 别名
    else:
        metrics['unknown_recall'] = 0.0
        metrics['unknown_acc'] = 0.0
    
    # 7. 已知类召回率
    if known_mask.sum() > 0:
        known_correct = (predictions[known_mask] == labels[known_mask]).sum()
        metrics['known_recall'] = known_correct / known_mask.sum()
    else:
        metrics['known_recall'] = 0.0
    
    # 8. H-score (调和平均数)
    # H-score = 2 * (known_acc * unknown_acc) / (known_acc + unknown_acc)
    known_acc = metrics['known_acc']
    unknown_acc = metrics['unknown_acc']
    
    if known_acc + unknown_acc > 0:
        metrics['h_score'] = 2 * (known_acc * unknown_acc) / (known_acc + unknown_acc)
    else:
        metrics['h_score'] = 0.0
    
    return metrics


def compute_detailed_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    is_unknown: np.ndarray
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """计算详细的 OSR 指标，包括混淆矩阵等
    
    Args:
        labels: 真实标签
        predictions: 预测标签
        scores: 预测分数
        is_unknown: 是否为未知类
        
    Returns:
        metrics: 指标字典
        details: 详细信息字典
    """
    # 基础指标
    metrics = compute_osr_metrics(labels, predictions, scores, is_unknown)
    
    # 详细信息
    details = {}
    
    # 已知类和未知类的分离统计
    known_mask = ~is_unknown.astype(bool)
    
    if known_mask.sum() > 0:
        # 已知类的混淆矩阵
        cm_known = confusion_matrix(
            labels[known_mask],
            predictions[known_mask]
        )
        details['confusion_matrix_known'] = cm_known
        
        # 每个类别的准确率
        class_acc = {}
        for class_id in np.unique(labels[known_mask]):
            mask = labels[known_mask] == class_id
            if mask.sum() > 0:
                acc = (predictions[known_mask][mask] == class_id).sum() / mask.sum()
                class_acc[int(class_id)] = float(acc)
        details['class_accuracy'] = class_acc
    
    # 分数分布统计
    max_scores = scores.max(axis=1)
    details['score_stats'] = {
        'known_mean': float(max_scores[known_mask].mean()) if known_mask.sum() > 0 else 0.0,
        'known_std': float(max_scores[known_mask].std()) if known_mask.sum() > 0 else 0.0,
        'unknown_mean': float(max_scores[~known_mask].mean()) if (~known_mask).sum() > 0 else 0.0,
        'unknown_std': float(max_scores[~known_mask].std()) if (~known_mask).sum() > 0 else 0.0,
    }
    
    return metrics, details


def compute_oscr(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    is_unknown: np.ndarray,
    fpr_threshold: float = 0.05
) -> float:
    """计算 OSCR (Open Set Classification Rate)
    
    在固定假阳率下，正确分类已知类的比率
    
    Args:
        labels: 真实标签
        predictions: 预测标签
        scores: 置信度分数
        is_unknown: 是否为未知类
        fpr_threshold: 假阳率阈值
        
    Returns:
        oscr: OSCR 值
    """
    known_mask = ~is_unknown.astype(bool)
    
    if known_mask.sum() == 0 or is_unknown.sum() == 0:
        return 0.0
    
    # 根据 FPR 阈值确定分数阈值
    sorted_scores = np.sort(scores[known_mask])
    threshold_idx = int(len(sorted_scores) * fpr_threshold)
    threshold = sorted_scores[threshold_idx] if threshold_idx < len(sorted_scores) else 0.0
    
    # 预测为已知类（分数 >= 阈值）
    predicted_known = scores >= threshold
    
    # 在已知类中，正确分类的数量
    correct_known = (predicted_known & known_mask & (predictions == labels)).sum()
    
    # OSCR = 正确分类的已知类 / 总已知类
    oscr = correct_known / known_mask.sum()
    
    return float(oscr)
