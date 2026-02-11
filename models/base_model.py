"""
基础分类器模型
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class BaseClassifier(nn.Module, ABC):
    """基础分类器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        self.num_classes = config.get('num_classes', 3)
        self.embedding_dim = config.get('embedding_dim', 64)
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入数据
            
        Returns:
            logits: (batch_size, num_classes) 分类 logits
            features: (batch_size, embedding_dim) 特征嵌入
        """
        pass
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """获取特征嵌入
        
        Args:
            x: 输入数据
            
        Returns:
            features: (batch_size, embedding_dim)
        """
        _, features = self.forward(x)
        return features
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测类别
        
        Args:
            x: 输入数据
            
        Returns:
            predictions: (batch_size,) 预测类别
        """
        logits, _ = self.forward(x)
        return torch.argmax(logits, dim=1)
