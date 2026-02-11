"""
LPVG + MLP 分类器
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .base_model import BaseClassifier


class LPVG_MLP(BaseClassifier):
    """LPVG 图特征 + MLP 分类器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 模型配置
        """
        super().__init__(config)
        
        # 图特征维度（默认 11 个拓扑特征）
        self.input_dim = config.get('graph_features', {}).get('feature_dim', 11)
        
        # MLP 配置
        classifier_config = config.get('classifier', {})
        hidden_dims = classifier_config.get('hidden_dims', [256, 128, 64])
        dropout = classifier_config.get('dropout', 0.3)
        use_batchnorm = classifier_config.get('use_batchnorm', True)
        activation = classifier_config.get('activation', 'relu')
        
        # 构建 MLP
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 嵌入层（最后一个隐藏层作为特征嵌入）
        self.embedding_dim = hidden_dims[-1] if hidden_dims else self.input_dim
        
        # 分类头
        self.classifier_head = nn.Linear(self.embedding_dim, self.num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            x: (batch_size, input_dim) 图特征
            
        Returns:
            logits: (batch_size, num_classes) 分类 logits
            features: (batch_size, embedding_dim) 特征嵌入
        """
        # 特征提取
        features = self.feature_extractor(x)
        
        # 分类
        logits = self.classifier_head(features)
        
        return logits, features
    
    def get_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            'name': 'lpvg_mlp',
            'input_dim': self.input_dim,
            'embedding_dim': self.embedding_dim,
            'num_classes': self.num_classes,
            'config': self.config
        }
