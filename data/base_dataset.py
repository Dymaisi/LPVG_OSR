"""
开放集识别基础数据集类
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple, List, Optional
from abc import ABC, abstractmethod


class BaseOSRDataset(Dataset, ABC):
    """开放集识别基础数据集类
    
    与闭集识别的区别：
    1. 训练集只包含已知类
    2. 测试集包含已知类和未知类
    3. 提供开放集评估所需的标签信息
    """
    
    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        """
        Args:
            config: 数据集配置
            split: 数据集划分 ('train', 'val', 'test')
        """
        self.config = config
        self.split = split
        
        # 开放集配置
        self.known_classes = config['known_classes']
        self.unknown_classes = config.get('unknown_classes', [])
        self.num_known_classes = len(self.known_classes)
        
        # 数据存储
        self.signals: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.is_unknown: Optional[np.ndarray] = None  # 是否为未知类
        
        # LPVG 图缓存
        self.graphs: Optional[List] = None
        self.graph_features: Optional[np.ndarray] = None
        
    @abstractmethod
    def load_data(self) -> None:
        """加载数据"""
        pass
    
    def build_lpvg_graphs(self, m: int = 1, use_cache: bool = True) -> None:
        """构建 LPVG 图
        
        Args:
            m: 穿透参数
            use_cache: 是否使用缓存
        """
        from .lpvg_builder import LPVGBuilder
        
        builder = LPVGBuilder(m=m, use_cache=use_cache)
        self.graphs = builder.build_batch(self.signals)
        
    def extract_graph_features(self) -> np.ndarray:
        """提取图特征
        
        Returns:
            graph_features: (N, feature_dim) 图特征矩阵
        """
        from .lpvg_builder import extract_graph_features
        
        if self.graphs is None:
            raise ValueError("请先调用 build_lpvg_graphs() 构建图")
            
        features = []
        for G in self.graphs:
            feat = extract_graph_features(G)
            features.append(feat)
            
        self.graph_features = np.array(features)
        return self.graph_features
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.signals) if self.signals is not None else 0
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """获取单个样本
        
        Returns:
            signal: 信号数据
            label: 类别标签（对于未知类，标签为 -1）
            is_unknown: 是否为未知类 (0/1)
        """
        signal = torch.FloatTensor(self.signals[index])
        label = int(self.labels[index])
        is_unknown = int(self.is_unknown[index])
        
        return signal, label, is_unknown
    
    def get_known_class_mask(self) -> np.ndarray:
        """获取已知类的mask
        
        Returns:
            mask: (N,) 布尔数组，True 表示已知类
        """
        return ~self.is_unknown.astype(bool)
    
    def get_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        known_mask = self.get_known_class_mask()
        
        return {
            'split': self.split,
            'total_samples': len(self),
            'known_samples': known_mask.sum(),
            'unknown_samples': (~known_mask).sum(),
            'known_classes': self.known_classes,
            'unknown_classes': self.unknown_classes,
            'num_known_classes': self.num_known_classes,
            'signal_length': self.signals.shape[-1] if self.signals is not None else 0,
            'has_graphs': self.graphs is not None,
            'has_features': self.graph_features is not None
        }
    
    def get_class_distribution(self) -> Dict[int, int]:
        """获取类别分布
        
        Returns:
            distribution: {class_id: count}
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
