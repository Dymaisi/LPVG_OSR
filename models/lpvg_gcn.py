"""
LPVG + GCN 图神经网络分类器 (简化版，不依赖 PyTorch Geometric)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List
import numpy as np


class GraphConvolution(nn.Module):
    """简单的图卷积层 (不依赖 PyG)"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            bias: 是否使用偏置
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: (num_nodes, in_features) 节点特征
            adj: (num_nodes, num_nodes) 归一化邻接矩阵
            
        Returns:
            out: (num_nodes, out_features) 输出特征
        """
        # H' = A * H * W
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support) if adj.is_sparse else torch.mm(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class LPVG_GCN(nn.Module):
    """LPVG + GCN 分类器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        self.num_classes = config.get('num_classes', 3)
        self.embedding_dim = config.get('embedding_dim', 64)
        
        # GNN 配置
        gnn_config = config.get('gnn', {})
        hidden_dims = gnn_config.get('hidden_dims', [64, 64, 32])
        dropout = gnn_config.get('dropout', 0.3)
        
        # 节点初始特征维度 (使用节点的信号值作为特征)
        self.node_feature_dim = 1  # 每个节点只有信号值一个特征
        
        # 构建 GCN 层
        self.gc_layers = nn.ModuleList()
        prev_dim = self.node_feature_dim
        
        for hidden_dim in hidden_dims:
            self.gc_layers.append(GraphConvolution(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.dropout = nn.Dropout(dropout)
        
        # 图池化后的特征维度
        self.graph_feature_dim = hidden_dims[-1] if hidden_dims else self.node_feature_dim
        
        # 分类器
        classifier_config = config.get('classifier', {})
        classifier_hidden = classifier_config.get('hidden_dims', [128, 64])
        
        # MLP 分类器
        mlp_layers = []
        prev_dim = self.graph_feature_dim
        
        for hidden_dim in classifier_hidden:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*mlp_layers)
        self.embedding_dim = classifier_hidden[-1] if classifier_hidden else self.graph_feature_dim
        
        # 分类头
        self.classifier_head = nn.Linear(self.embedding_dim, self.num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            node_features: (batch_size, num_nodes, node_feature_dim) 节点特征
            adj_matrix: (batch_size, num_nodes, num_nodes) 邻接矩阵
            
        Returns:
            logits: (batch_size, num_classes) 分类 logits
            features: (batch_size, embedding_dim) 特征嵌入
        """
        batch_size = node_features.size(0)
        graph_embeddings = []
        
        # 对每个图分别处理
        for i in range(batch_size):
            x = node_features[i]  # (num_nodes, node_feature_dim)
            adj = adj_matrix[i]   # (num_nodes, num_nodes)
            
            # 归一化邻接矩阵 (添加自环 + 度归一化)
            adj_normalized = self._normalize_adjacency(adj)
            
            # GCN 层
            for idx, gc_layer in enumerate(self.gc_layers):
                x = gc_layer(x, adj_normalized)
                if idx < len(self.gc_layers) - 1:
                    x = F.relu(x)
                    x = self.dropout(x)
            
            # 图池化 (全局平均池化)
            graph_emb = torch.mean(x, dim=0)  # (graph_feature_dim,)
            graph_embeddings.append(graph_emb)
        
        # 堆叠成批次
        graph_embeddings = torch.stack(graph_embeddings)  # (batch_size, graph_feature_dim)
        
        # MLP 特征提取
        features = self.feature_extractor(graph_embeddings)  # (batch_size, embedding_dim)
        
        # 分类
        logits = self.classifier_head(features)  # (batch_size, num_classes)
        
        return logits, features
    
    def _normalize_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        """归一化邻接矩阵
        
        A_norm = D^(-1/2) * (A + I) * D^(-1/2)
        
        Args:
            adj: (num_nodes, num_nodes) 邻接矩阵
            
        Returns:
            adj_normalized: 归一化后的邻接矩阵
        """
        # 添加自环
        num_nodes = adj.size(0)
        adj = adj + torch.eye(num_nodes, device=adj.device)
        
        # 计算度矩阵
        degree = adj.sum(dim=1)
        
        # D^(-1/2)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        
        # D^(-1/2) * A * D^(-1/2)
        adj_normalized = adj * degree_inv_sqrt.view(-1, 1) * degree_inv_sqrt.view(1, -1)
        
        return adj_normalized
    
    def predict(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """预测类别
        
        Args:
            node_features: 节点特征
            adj_matrix: 邻接矩阵
            
        Returns:
            predictions: (batch_size,) 预测类别
        """
        logits, _ = self.forward(node_features, adj_matrix)
        return torch.argmax(logits, dim=1)
    
    def get_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            'name': 'lpvg_gcn',
            'node_feature_dim': self.node_feature_dim,
            'graph_feature_dim': self.graph_feature_dim,
            'embedding_dim': self.embedding_dim,
            'num_classes': self.num_classes,
            'config': self.config
        }
