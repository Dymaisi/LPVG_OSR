"""
LPVG-EP-DyGAT 模型 - 完整实现
严格按照guide.md理论实现所有组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from typing import Dict, Optional


class LPVG_EP_DyGAT(nn.Module):
    """
    LPVG-EP-DyGAT: 基于证据原型的动态图注意力网络
    
    完整实现guide.md中的三大模块：
    1. LPVG拓扑映射 (预处理阶段)
    2. DyGAT深层特征提取 (带初始残差)
    3. EP证据原型分类 (基于RBF距离)
    
    关键改进：
    - ✅ 类原型维护
    - ✅ RBF证据生成 (距离度量)
    - ✅ 初始残差连接 (防过平滑)
    - ✅ 支持原型聚类损失
    """
    
    def __init__(
        self,
        node_feature_dim: int = 4,
        hidden_dims: list = [64, 64, 32],
        num_heads: int = 4,
        num_classes: int = 3,
        dropout: float = 0.3,
        pooling: str = 'mean',
        alpha_residual: float = 0.1,  # 初始残差权重
        rbf_sigma: float = 7.0        # RBF带宽（平衡证据强度）
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.pooling = pooling
        self.alpha_residual = alpha_residual
        # rbf_sigma将通过register_buffer注册为固定值（不参与梯度）
        self.feature_dim = hidden_dims[2]
        
        # =============== DyGAT组件 ===============
        # GAT层 (PyG原生，支持真正的batch)
        self.conv1 = GATConv(
            node_feature_dim, 
            hidden_dims[0] // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        self.conv2 = GATConv(
            hidden_dims[0],
            hidden_dims[1] // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        self.conv3 = GATConv(
            hidden_dims[1],
            hidden_dims[2],
            heads=1,
            dropout=dropout,
            concat=False
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # LayerNorm用于稳定特征（替代L2归一化）
        self.layer_norm = nn.LayerNorm(hidden_dims[2])
        
        # 初始特征投影（用于残差连接）
        self.initial_projection = nn.Linear(node_feature_dim, hidden_dims[2])
        
        # =============== EP组件 ===============
        # 类原型 (Prototypes) - 可学习参数
        # 形状: (num_classes, feature_dim)
        # 初始化到与LayerNorm特征相同尺度（~1.0）
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, hidden_dims[2]) * 1.0
        )
        
        # RBF温度参数（固定值，避免训练时漂移）
        # sigma=7.0是根据LayerNorm后特征范围调优的平衡值
        self.register_buffer('rbf_sigma', torch.tensor(rbf_sigma))
    
    def forward(self, x, edge_index, batch):
        """
        完整的LPVG-EP-DyGAT前向传播
        
        Args:
            x: (total_nodes, node_feat_dim) - 所有图的节点特征拼接
            edge_index: (2, total_edges) - 所有图的边索引拼接
            batch: (total_nodes,) - 节点到图的映射
        
        Returns:
            outputs: 包含所有中间结果的字典
        """
        # =============== 保存初始特征 (关键!) ===============
        # guide.md 3.3.3: 初始残差连接需要H^(0)
        h0_projected = self.initial_projection(x)  # (total_nodes, feature_dim)
        if self.pooling == 'mean':
            h0_graph = global_mean_pool(h0_projected, batch)  # (batch_size, feature_dim)
        else:
            h0_graph = global_max_pool(h0_projected, batch)
        
        # =============== DyGAT特征提取 ===============
        # GAT层1
        x1 = self.conv1(x, edge_index)
        x1 = F.elu(x1)
        x1 = self.dropout(x1)
        
        # GAT层2
        x2 = self.conv2(x1, edge_index)
        x2 = F.elu(x2)
        x2 = self.dropout(x2)
        
        # GAT层3（最终特征）
        x3 = self.conv3(x2, edge_index)
        x3 = F.elu(x3)
        
        # 全局池化 (节点级别 -> 图级别)
        if self.pooling == 'mean':
            graph_embedding = global_mean_pool(x3, batch)
        else:
            graph_embedding = global_max_pool(x3, batch)
        
        # =============== 初始残差连接 ===============
        # guide.md 3.3.3: H^(l+1) = (1-α)·H^(l) + α·H^(0)
        # 这防止深层网络过平滑，保持特征可区分性
        graph_embedding = (1 - self.alpha_residual) * graph_embedding + \
                         self.alpha_residual * h0_graph
        
        # LayerNorm稳定特征（避免爆炸，但保留相对距离信息）
        graph_embedding = self.layer_norm(graph_embedding)
        
        # =============== EP证据生成 (RBF距离) ===============
        # guide.md 3.4.1: e_ik = exp(-||z_i - c_k||^2 / (2σ^2))
        sigma = self.rbf_sigma  # 使用固定sigma，避免训练漂移
        
        # 计算特征与原型的欧氏距离（无归一化）
        # graph_embedding: (batch_size, feature_dim)
        # prototypes: (num_classes, feature_dim)
        distances = torch.cdist(
            graph_embedding.unsqueeze(1),  # (batch_size, 1, feature_dim)
            self.prototypes.unsqueeze(0),  # (1, num_classes, feature_dim)
            p=2  # 欧氏距离
        ).squeeze(1)  # (batch_size, num_classes)
        
        # RBF核函数生成证据（数值稳定版）
        # 防止exp(-x)在x很大时下溢为0
        scaled_distances = distances ** 2 / (2 * sigma ** 2)
        scaled_distances = torch.clamp(scaled_distances, max=50.0)  # 增大范围
        evidence = torch.exp(-scaled_distances)
        evidence = torch.clamp(evidence, min=1e-10, max=1e3)  # 调整范围
        
        # =============== Dirichlet参数化 ===============
        # guide.md 3.4.2
        alpha = evidence + 1.0  # α = e + 1
        S = torch.sum(alpha, dim=1, keepdim=True)  # S = Σα_k
        belief = evidence / S  # b_k = e_k / S (信念质量)
        uncertainty = self.num_classes / S  # u = K / S (认知不确定性)
        
        return {
            'graph_embedding': graph_embedding,      # 最终特征向量（无归一化）
            'h0_graph': h0_graph,                    # 初始特征（用于残差）
            'distances': distances,                   # 到原型的距离
            'evidence': evidence,                     # 证据向量
            'alpha': alpha,                          # Dirichlet参数
            'S': S,                                  # 证据强度
            'belief': belief,                        # 信念质量
            'uncertainty': uncertainty,              # 不确定性
            'prototypes': self.prototypes            # 类原型（用于损失计算）
        }


def create_lpvg_ep_dygat(config: Dict) -> LPVG_EP_DyGAT:
    """
    工厂函数 - 创建完整的LPVG-EP-DyGAT模型
    
    Args:
        config: 配置字典，包含所有超参数
    
    Returns:
        完整实现的LPVG-EP-DyGAT模型
    """
    return LPVG_EP_DyGAT(
        node_feature_dim=config.get('node_feature_dim', 4),
        hidden_dims=config.get('hidden_dims', [64, 64, 32]),
        num_heads=config.get('num_heads', 4),
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.3),
        pooling=config.get('pooling', 'mean'),
        alpha_residual=config.get('alpha_residual', 0.1),
        rbf_sigma=config.get('rbf_sigma', 1.0)
    )


# 向后兼容（保留旧接口）
LPVG_GNN_PyG = LPVG_EP_DyGAT
create_lpvg_gnn_pyg = create_lpvg_ep_dygat
