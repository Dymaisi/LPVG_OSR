"""
LPVG-EP-DyGAT 完整模型
严格按照论文实现的端到端开放集故障诊断框架

论文架构：
输入信号 → LPVG 图构建 → DyGAT 特征提取（带初始残差）→ 证据原型分类器 → 开放集决策
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import numpy as np
import networkx as nx

try:
    from .dygat_residual import DyGATWithInitialResidual
    from .evidential_classifier import EvidentialPrototypicalClassifier, open_set_prediction
except ImportError:
    from dygat_residual import DyGATWithInitialResidual
    from evidential_classifier import EvidentialPrototypicalClassifier, open_set_prediction


class LPVGEPDyGAT(nn.Module):
    """
    完整的 LPVG-EP-DyGAT 模型（论文完整实现）
    
    三大模块：
    1. LPVG 图构建（在数据预处理阶段完成）
    2. DyGAT 特征提取器（带初始残差连接）
    3. 证据原型分类器（距离度量 + RBF 证据映射）
    """
    
    def __init__(
        self,
        # 输入参数
        node_feature_dim: int = 4,  # [value, local_diff, local_mean, energy]
        
        # DyGAT 参数
        dygat_hidden_dims: List[int] = [64, 64, 32],
        dygat_num_heads: int = 8,
        dygat_dropout: float = 0.3,
        lambda_residual: float = 2.0,
        alpha_residual: float = 0.1,
        use_dynamic_graph: bool = True,
        top_k: int = 10,
        mu: float = 0.5,
        
        # 证据分类器参数
        num_classes: int = 3,
        sigma: float = 1.0,
        learn_prototypes: bool = True,
        
        # 全局池化方式
        global_pooling: str = 'mean'  # 'mean', 'max', 'sum'
    ):
        """
        Args:
            node_feature_dim: 节点特征维度（LPVG 图的初始特征）
            dygat_hidden_dims: DyGAT 各层隐藏维度
            dygat_num_heads: 注意力头数
            dygat_dropout: Dropout 率
            lambda_residual: 残差衰减参数 λ
            use_dynamic_graph: 是否使用动态图构建
            top_k: 动态图 top-k
            mu: LPVG 与动态图融合权重
            num_classes: 已知故障类别数
            sigma: RBF 核缩放参数
            learn_prototypes: 原型是否可学习
            global_pooling: 全局池化方式
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.num_classes = num_classes
        self.global_pooling = global_pooling
        
        # 模块 1: DyGAT 特征提取器
        self.dygat = DyGATWithInitialResidual(
            input_dim=node_feature_dim,
            hidden_dims=dygat_hidden_dims,
            num_heads=dygat_num_heads,
            dropout=dygat_dropout,
            lambda_residual=lambda_residual,
            alpha_residual=alpha_residual,
            use_dynamic_graph=use_dynamic_graph,
            top_k=top_k,
            mu=mu
        )
        
        # 模块 2: 证据原型分类器
        graph_feature_dim = dygat_hidden_dims[-1]  # DyGAT 输出维度
        
        self.evidential_classifier = EvidentialPrototypicalClassifier(
            feature_dim=graph_feature_dim,
            num_classes=num_classes,
            sigma=sigma,
            learn_prototypes=learn_prototypes
        )
    
    def global_graph_pooling(
        self, 
        node_embeddings: torch.Tensor,
        batch_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        全局图池化（将节点嵌入聚合为图级表示）
        
        Args:
            node_embeddings: (num_nodes, embedding_dim) 节点嵌入
            batch_mask: (num_nodes,) 批次掩码（用于批处理）
            
        Returns:
            graph_embedding: (embedding_dim,) 图级嵌入
        """
        if self.global_pooling == 'mean':
            return torch.mean(node_embeddings, dim=0)
        elif self.global_pooling == 'max':
            return torch.max(node_embeddings, dim=0)[0]
        elif self.global_pooling == 'sum':
            return torch.sum(node_embeddings, dim=0)
        else:
            raise ValueError(f"Unknown pooling: {self.global_pooling}")
    
    def forward(
        self,
        node_features: torch.Tensor,
        adj_lpvg: torch.Tensor,
        return_embeddings: bool = False,
        return_distances: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            node_features: (num_nodes, node_feature_dim) 节点特征矩阵 H^(0)
            adj_lpvg: (num_nodes, num_nodes) LPVG 邻接矩阵
            return_embeddings: 是否返回节点嵌入
            return_distances: 是否返回距离信息
            
        Returns:
            outputs: 包含所有中间结果和最终预测的字典
        """
        # Step 1: DyGAT 特征提取（带初始残差）
        node_embeddings = self.dygat(node_features, adj_lpvg)  # (N, hidden_dims[-1])
        
        # Step 2: 全局图池化
        graph_embedding = self.global_graph_pooling(node_embeddings)  # (hidden_dims[-1],)
        graph_embedding = graph_embedding.unsqueeze(0)  # (1, hidden_dims[-1])
        
        # Step 3: 证据原型分类
        evidential_outputs = self.evidential_classifier(
            graph_embedding, 
            return_distances=return_distances
        )
        
        # 组合输出
        outputs = evidential_outputs
        outputs['graph_embedding'] = graph_embedding.squeeze(0)
        
        if return_embeddings:
            outputs['node_embeddings'] = node_embeddings
        
        return outputs
    
    def forward_batch(
        self,
        batch_node_features,
        batch_adj_lpvg,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        批量前向传播
        
        Args:
            batch_node_features: (batch_size, num_nodes, node_feature_dim)
            batch_adj_lpvg: (batch_size, num_nodes, num_nodes)
            return_embeddings: 是否返回嵌入
            
        Returns:
            outputs: 批量输出字典
        """
        if isinstance(batch_node_features, (list, tuple)):
            batch_size = len(batch_node_features)
        else:
            batch_size = batch_node_features.shape[0]
        
        # 存储批量结果
        all_graph_embeddings = []
        all_evidences = []
        all_alphas = []
        all_beliefs = []
        all_uncertainties = []
        
        # 逐样本处理（因为每个图的节点数可能不同）
        for i in range(batch_size):
            node_features = batch_node_features[i]
            adj_lpvg = batch_adj_lpvg[i]
            outputs = self.forward(
                node_features,
                adj_lpvg,
                return_embeddings=False,
                return_distances=False
            )
            
            all_graph_embeddings.append(outputs['graph_embedding'])
            all_evidences.append(outputs['evidence'])
            all_alphas.append(outputs['alpha'])
            all_beliefs.append(outputs['belief'])
            all_uncertainties.append(outputs['uncertainty'])
        
        # 堆叠结果
        batch_outputs = {
            'graph_embedding': torch.stack(all_graph_embeddings, dim=0),  # (B, D)
            'evidence': torch.cat(all_evidences, dim=0),  # (B, K)
            'alpha': torch.cat(all_alphas, dim=0),  # (B, K)
            'S': torch.sum(torch.cat(all_alphas, dim=0), dim=1, keepdim=True),  # (B, 1)
            'belief': torch.cat(all_beliefs, dim=0),  # (B, K)
            'uncertainty': torch.cat(all_uncertainties, dim=0),  # (B, 1)
            'expected_prob': torch.cat(all_alphas, dim=0) / torch.sum(torch.cat(all_alphas, dim=0), dim=1, keepdim=True)
        }
        
        return batch_outputs
    
    def predict(
        self,
        node_features: torch.Tensor,
        adj_lpvg: torch.Tensor,
        uncertainty_threshold: float = 0.5
    ) -> Tuple[int, bool, float]:
        """
        单样本预测（论文 Section 3.6）
        
        Args:
            node_features: (num_nodes, node_feature_dim)
            adj_lpvg: (num_nodes, num_nodes)
            uncertainty_threshold: 不确定性阈值 τ
            
        Returns:
            prediction: 预测类别（-1 表示未知）
            is_unknown: 是否为未知类
            uncertainty: 不确定性分数
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(node_features, adj_lpvg)
            
            predictions, is_unknown_mask = open_set_prediction(
                outputs, 
                uncertainty_threshold
            )
            
            prediction = predictions.item() if predictions.dim() == 0 else predictions[0].item()
            is_unknown = is_unknown_mask.item() if is_unknown_mask.dim() == 0 else is_unknown_mask[0].item()
            uncertainty = outputs['uncertainty'].item() if outputs['uncertainty'].dim() == 0 else outputs['uncertainty'][0].item()
        
        return prediction, is_unknown, uncertainty
    
    def get_prototypes(self) -> torch.Tensor:
        """获取类原型"""
        return self.evidential_classifier.prototypes
    
    def set_prototypes(self, prototypes: torch.Tensor):
        """设置类原型"""
        self.evidential_classifier.prototypes.data = prototypes


def create_lpvg_ep_dygat(config: Dict) -> LPVGEPDyGAT:
    """
    根据配置创建模型
    
    Args:
        config: 配置字典
        
    Returns:
        model: LPVG-EP-DyGAT 模型
    """
    model = LPVGEPDyGAT(
        node_feature_dim=config.get('node_feature_dim', 4),
        dygat_hidden_dims=config.get('dygat_hidden_dims', [64, 64, 32]),
        dygat_num_heads=config.get('dygat_num_heads', 8),
        dygat_dropout=config.get('dygat_dropout', 0.3),
        lambda_residual=config.get('lambda_residual', 2.0),
        alpha_residual=config.get('alpha_residual', 0.1),
        use_dynamic_graph=config.get('use_dynamic_graph', True),
        top_k=config.get('top_k', 10),
        mu=config.get('mu', 0.5),
        num_classes=config.get('num_classes', 3),
        sigma=config.get('sigma', 1.0),
        learn_prototypes=config.get('learn_prototypes', True),
        global_pooling=config.get('global_pooling', 'mean')
    )
    
    return model


if __name__ == "__main__":
    print("=" * 80)
    print("LPVG-EP-DyGAT 完整模型测试")
    print("=" * 80)
    
    # 测试参数
    num_nodes = 256
    node_feature_dim = 4
    num_classes = 3
    batch_size = 4
    
    # 生成测试数据
    node_features = torch.randn(num_nodes, node_feature_dim)
    adj_lpvg = torch.rand(num_nodes, num_nodes)
    adj_lpvg = (adj_lpvg > 0.95).float()
    adj_lpvg = (adj_lpvg + adj_lpvg.T) / 2
    
    # 创建模型
    config = {
        'node_feature_dim': node_feature_dim,
        'dygat_hidden_dims': [64, 64, 32],
        'dygat_num_heads': 8,
        'dygat_dropout': 0.3,
        'lambda_residual': 2.0,
        'alpha_residual': 0.1,
        'num_classes': num_classes,
        'sigma': 1.0,
        'global_pooling': 'mean'
    }
    
    model = create_lpvg_ep_dygat(config)
    
    print(f"\n模型配置:")
    print(f"  节点特征维度: {node_feature_dim}")
    print(f"  已知类别数: {num_classes}")
    print(f"  DyGAT 层数: {len(config['dygat_hidden_dims'])}")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 单样本前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(node_features, adj_lpvg, return_embeddings=True)
    
    print(f"\n单样本输出:")
    print(f"  图嵌入: {outputs['graph_embedding'].shape}")
    print(f"  证据量: {outputs['evidence'].shape}")
    print(f"  信念: {outputs['belief'].shape}")
    print(f"  不确定性: {outputs['uncertainty'].item():.4f}")
    
    # 单样本预测
    prediction, is_unknown, uncertainty = model.predict(
        node_features, 
        adj_lpvg, 
        uncertainty_threshold=0.5
    )
    
    print(f"\n单样本预测:")
    print(f"  预测类别: {prediction}")
    print(f"  是否未知: {is_unknown}")
    print(f"  不确定性: {uncertainty:.4f}")
    
    # 批量前向传播
    batch_node_features = torch.randn(batch_size, num_nodes, node_feature_dim)
    batch_adj_lpvg = adj_lpvg.unsqueeze(0).repeat(batch_size, 1, 1)
    
    with torch.no_grad():
        batch_outputs = model.forward_batch(batch_node_features, batch_adj_lpvg)
    
    print(f"\n批量输出:")
    print(f"  批次大小: {batch_size}")
    print(f"  图嵌入: {batch_outputs['graph_embedding'].shape}")
    print(f"  证据量: {batch_outputs['evidence'].shape}")
    print(f"  不确定性: {batch_outputs['uncertainty'].squeeze()}")
    
    print("\n✅ LPVG-EP-DyGAT 完整模型测试通过!")
