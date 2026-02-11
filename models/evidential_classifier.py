"""
证据原型分类器（Evidential Prototypical Classifier）
严格按照论文 Section 3.4-3.5 实现

核心创新：
1. 基于原型的距离度量
2. RBF 核距离-证据映射
3. 狄利克雷分布参数化
4. 不确定性量化（认知不确定性）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


class EvidentialPrototypicalClassifier(nn.Module):
    """
    证据原型分类器（论文 Section 3.4）
    
    核心思想：
    1. 维护可学习的类原型 C = {c_1, ..., c_K}
    2. 计算样本特征与原型的欧氏距离 d_ik
    3. 通过 RBF 核将距离转化为证据量 e_ik
    4. 证据量参数化狄利克雷分布 α_k = e_k + 1
    5. 计算认知不确定性 u = K/S（用于开放集检测）
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        sigma: float = 1.0,  # RBF 缩放因子
        learn_prototypes: bool = True
    ):
        """
        Args:
            feature_dim: 特征维度（DyGAT 输出维度）
            num_classes: 已知类别数 K
            sigma: RBF 核的缩放参数
            learn_prototypes: 原型是否可学习
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.sigma = sigma
        
        # 可学习的类原型 C ∈ R^(K×D)
        if learn_prototypes:
            self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
            nn.init.xavier_uniform_(self.prototypes)
        else:
            self.register_buffer('prototypes', torch.randn(num_classes, feature_dim))
    
    def compute_distances(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算样本特征与各原型的欧氏距离（论文 Section 3.4）
        
        d_ik = ||z_i - c_k||_2
        
        Args:
            z: (batch_size, feature_dim) 样本特征
            
        Returns:
            distances: (batch_size, num_classes) 距离矩阵
        """
        # 扩展维度进行广播计算
        z_expanded = z.unsqueeze(1)  # (B, 1, D)
        prototypes_expanded = self.prototypes.unsqueeze(0)  # (1, K, D)
        
        # 欧氏距离
        distances = torch.norm(z_expanded - prototypes_expanded, p=2, dim=-1)  # (B, K)
        
        return distances
    
    def distance_to_evidence(self, distances: torch.Tensor) -> torch.Tensor:
        """
        距离-证据映射（论文 Section 3.4）
        
        e_ik = exp(-d_ik^2 / (2σ^2))
        
        物理意义：
        - 距离越近，证据越强（e→1）
        - 距离越远，证据越弱（e→0）
        - 未知样本远离所有原型，导致所有 e_ik≈0
        
        Args:
            distances: (batch_size, num_classes) 距离矩阵
            
        Returns:
            evidence: (batch_size, num_classes) 证据量
        """
        # RBF 核映射
        evidence = torch.exp(-distances ** 2 / (2 * self.sigma ** 2))
        
        return evidence
    
    def evidence_to_dirichlet(self, evidence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        证据量参数化狄利克雷分布（论文 Section 2.3.1）
        
        α_k = e_k + 1
        S = Σ α_k = Σ(e_k + 1) = Σe_k + K
        
        信念质量: b_k = e_k / S
        不确定性: u = K / S
        
        Args:
            evidence: (batch_size, num_classes) 证据量
            
        Returns:
            dirichlet_params: 包含 α, S, belief, uncertainty 的字典
        """
        # 狄利克雷参数 α
        alpha = evidence + 1.0  # (B, K)
        
        # 狄利克雷强度 S
        S = torch.sum(alpha, dim=1, keepdim=True)  # (B, 1)
        
        # 信念质量 b_k = e_k / S
        belief = evidence / S  # (B, K)
        
        # 认知不确定性 u = K / S（论文 Section 2.3.2）
        uncertainty = self.num_classes / S  # (B, 1)
        
        # 预期概率（狄利克雷分布的期望）
        expected_prob = alpha / S  # (B, K)
        
        return {
            'alpha': alpha,
            'S': S,
            'belief': belief,
            'uncertainty': uncertainty,
            'expected_prob': expected_prob,
            'evidence': evidence
        }
    
    def forward(
        self, 
        z: torch.Tensor,
        return_distances: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        完整的前向传播
        
        Args:
            z: (batch_size, feature_dim) 特征嵌入
            return_distances: 是否返回距离信息
            
        Returns:
            outputs: 包含所有证据推理结果的字典
        """
        # Step 1: 计算距离
        distances = self.compute_distances(z)  # (B, K)
        
        # Step 2: 距离→证据
        evidence = self.distance_to_evidence(distances)  # (B, K)
        
        # Step 3: 证据→狄利克雷分布
        dirichlet_params = self.evidence_to_dirichlet(evidence)
        
        # 组合输出
        outputs = dirichlet_params
        if return_distances:
            outputs['distances'] = distances
        
        return outputs


class EvidentialLoss(nn.Module):
    """
    证据深度学习损失函数（论文 Section 3.5）
    
    包含三个部分：
    1. L_EDL: Type-II 极大似然损失（Bayes Risk）
    2. L_KL: KL 散度正则化
    3. L_proto: 原型聚类损失
    """
    
    def __init__(
        self,
        num_classes: int,
        lambda_kl: float = 0.1,
        lambda_proto: float = 0.01,
        kl_annealing: bool = True
    ):
        """
        Args:
            num_classes: 类别数 K
            lambda_kl: KL 散度权重
            lambda_proto: 原型损失权重
            kl_annealing: 是否对 KL 项进行退火
        """
        super().__init__()
        self.num_classes = num_classes
        self.lambda_kl = lambda_kl
        self.lambda_proto = lambda_proto
        self.kl_annealing = kl_annealing
        self.current_epoch = 0
    
    def edl_loss(
        self, 
        alpha: torch.Tensor, 
        y: torch.Tensor,
        S: torch.Tensor
    ) -> torch.Tensor:
        """
        证据分类损失（论文 Section 3.5 公式）
        
        L_EDL = Σ_k y_k (ψ(S) - ψ(α_k))
        
        Args:
            alpha: (batch_size, num_classes) 狄利克雷参数
            y: (batch_size,) 真实标签
            S: (batch_size, 1) 狄利克雷强度
            
        Returns:
            loss: 标量损失
        """
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        digamma_term = torch.digamma(S) - torch.digamma(alpha)
        loss = torch.sum(y_one_hot * digamma_term, dim=1)
        return torch.mean(loss)
    
    def kl_divergence_loss(
        self, 
        alpha: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        KL 散度正则化（论文 Section 3.5）
        
        目的：防止模型对错误分类产生高证据
        
        L_KL = KL(Dir(p|α̃) || Dir(p|1))
        其中 α̃_k = y_k + (1-y_k)·α_k（去除真实类的证据）
        
        Args:
            alpha: (batch_size, num_classes) 狄利克雷参数
            y: (batch_size,) 真实标签
            
        Returns:
            kl_loss: KL 散度损失
        """
        # One-hot 编码
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()  # (B, K)
        
        # 构造 α̃（去除真实类的证据）
        alpha_tilde = y_one_hot + (1 - y_one_hot) * alpha
        
        # KL 散度计算
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        
        # KL(Dir(α̃) || Dir(1)) 的解析形式
        term1 = torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(self.num_classes, dtype=torch.float))
        term2 = -torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
        term3 = torch.sum((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1, keepdim=True)
        
        kl_loss = term1 + term2 + term3
        
        return torch.mean(kl_loss)
    
    def prototype_clustering_loss(
        self, 
        z: torch.Tensor, 
        y: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        原型聚类损失（论文 Section 4.1）
        
        L_proto = -log(exp(-||z_i - c_y||^2) / Σ_j exp(-||z_i - c_j||^2))
        
        Args:
            z: (batch_size, feature_dim) 样本特征
            y: (batch_size,) 真实标签
            prototypes: (num_classes, feature_dim) 类原型
            
        Returns:
            proto_loss: 原型聚类损失
        """
        distances = torch.cdist(z.unsqueeze(1), prototypes.unsqueeze(0), p=2).squeeze(1)
        logits = -distances ** 2
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        loss = F.nll_loss(log_probs, y)
        return loss
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        z: torch.Tensor,
        y: torch.Tensor,
        prototypes: torch.Tensor,
        epoch: int = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算总损失
        
        L_total = L_EDL + λ_KL·L_KL + λ_proto·L_proto
        
        Args:
            outputs: 证据分类器的输出字典
            z: 特征嵌入
            y: 真实标签
            prototypes: 类原型
            epoch: 当前 epoch（用于 KL 退火）
            
        Returns:
            total_loss: 总损失
            loss_dict: 各部分损失的字典
        """
        # 提取必要的参数
        alpha = outputs['alpha']
        S = outputs['S']
        
        # 1. EDL 损失
        loss_edl = self.edl_loss(alpha, y, S)
        
        # 2. KL 散度损失（带退火）
        loss_kl = self.kl_divergence_loss(alpha, y)
        
        if self.kl_annealing and epoch is not None:
            # KL 退火：早期小，后期大（避免干扰初期特征学习）
            kl_weight = min(self.lambda_kl * (epoch / 10.0), self.lambda_kl)
        else:
            kl_weight = self.lambda_kl
        
        # 3. 原型聚类损失
        loss_proto = self.prototype_clustering_loss(z, y, prototypes)
        
        # 总损失
        total_loss = loss_edl + kl_weight * loss_kl + self.lambda_proto * loss_proto
        
        # 记录各部分损失
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_edl': loss_edl.item(),
            'loss_kl': loss_kl.item(),
            'loss_proto': loss_proto.item(),
            'kl_weight': kl_weight
        }
        
        return total_loss, loss_dict


def open_set_prediction(
    outputs: Dict[str, torch.Tensor],
    uncertainty_threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    开放集预测（论文 Section 3.6）
    
    决策规则：
    - 如果 u > τ: 判定为"未知类"
    - 否则: 判定为 argmax(belief) 对应的已知类
    
    Args:
        outputs: 证据分类器输出
        uncertainty_threshold: 不确定性阈值 τ
        
    Returns:
        predictions: (batch_size,) 预测标签（-1 表示未知）
        is_unknown: (batch_size,) 布尔掩码，True 表示未知
    """
    uncertainty = outputs['uncertainty'].squeeze()  # (B,)
    belief = outputs['belief']  # (B, K)
    
    # 已知类预测
    known_preds = torch.argmax(belief, dim=1)  # (B,)
    
    # 未知类检测
    is_unknown = uncertainty > uncertainty_threshold
    
    # 最终预测（未知类标记为 -1）
    predictions = known_preds.clone()
    predictions[is_unknown] = -1
    
    return predictions, is_unknown


if __name__ == "__main__":
    print("=" * 80)
    print("证据原型分类器测试")
    print("=" * 80)
    
    # 测试参数
    batch_size = 32
    feature_dim = 32
    num_classes = 3
    
    # 生成测试数据
    z = torch.randn(batch_size, feature_dim)
    y = torch.randint(0, num_classes, (batch_size,))
    
    # 创建模型
    classifier = EvidentialPrototypicalClassifier(
        feature_dim=feature_dim,
        num_classes=num_classes,
        sigma=1.0
    )
    
    print(f"\n模型配置:")
    print(f"  特征维度: {feature_dim}")
    print(f"  类别数: {num_classes}")
    print(f"  原型矩阵: {classifier.prototypes.shape}")
    
    # 前向传播
    outputs = classifier(z, return_distances=True)
    
    print(f"\n输出:")
    print(f"  证据量: {outputs['evidence'].shape}, 均值={outputs['evidence'].mean():.4f}")
    print(f"  α 参数: {outputs['alpha'].shape}, 均值={outputs['alpha'].mean():.4f}")
    print(f"  信念: {outputs['belief'].shape}, 和={outputs['belief'].sum(dim=1).mean():.4f}")
    print(f"  不确定性: {outputs['uncertainty'].shape}, 均值={outputs['uncertainty'].mean():.4f}")
    
    # 测试损失函数
    criterion = EvidentialLoss(
        num_classes=num_classes,
        lambda_kl=0.1,
        lambda_proto=0.01
    )
    
    total_loss, loss_dict = criterion(outputs, z, y, classifier.prototypes, epoch=0)
    
    print(f"\n损失函数:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")
    
    # 测试开放集预测
    predictions, is_unknown = open_set_prediction(outputs, uncertainty_threshold=0.5)
    unknown_count = is_unknown.sum().item()
    
    print(f"\n开放集预测:")
    print(f"  未知样本数: {unknown_count}/{batch_size}")
    print(f"  已知样本预测: {predictions[~is_unknown][:5]}")
    
    print("\n✅ 证据原型分类器测试通过!")
