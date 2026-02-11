"""
CWRU 开放集数据集
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any
from .base_dataset import BaseOSRDataset


class CWRUOpenSet(BaseOSRDataset):
    """CWRU 轴承数据集 - 开放集版本
    
    类别定义：
    - 0: Normal (健康)
    - 1: Inner Race Fault (内圈故障)
    - 2: Outer Race Fault (外圈故障)
    - 3: Ball Fault (滚珠故障)
    """
    
    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        """
        Args:
            config: 数据集配置
            split: 'train', 'val', 'test'
        """
        super().__init__(config, split)
        self.data_root = Path(config['data_root'])
        self.signal_length = config.get('signal_length', 2048)
        self.overlap = config.get('overlap', 0.5)
        self.sampling_rate = config.get('sampling_rate', 12000)
        
        # 加载数据
        self.load_data()
        
    def load_data(self) -> None:
        """加载 CWRU 数据"""
        try:
            # 使用本地的 cwru_dataset（LPVG_OSR项目内部）
            try:
                from cwru_dataset import CWRUDataset
            except ImportError:
                from data.cwru_dataset import CWRUDataset
            
            # 构建配置
            target_rpm = self.config.get('target_rpm', self.config.get('rpm', '1797'))
            cwru_config = {
                'fault_type': self.config.get('fault_type', '12DriveEndFault'),
                'rpm': str(target_rpm),
                'is_source': True,
                'known_classes': self.known_classes,
                'unknown_classes': self.unknown_classes,
                'preprocessing': 'raw',
                'normalization': self.config.get('preprocessing', {}).get('normalization', 'zscore'),
                'sequence_length': self.signal_length,
                'train_ratio': self.config.get('train_ratio', 0.7),
            }
            
            dataset = CWRUDataset(cwru_config)
            
            # 根据 split 获取数据
            if self.split == 'train':
                self.signals = dataset.X_train.numpy()
                self.labels = dataset.y_train.numpy()
            elif self.split == 'val':
                # CWRU 原始没有 val，从 test 中分一部分
                X_test = dataset.X_test if dataset.X_test is not None else dataset.X_train
                y_test = dataset.y_test if dataset.y_test is not None else dataset.y_train
                n_val = len(X_test) // 2
                self.signals = X_test[:n_val].numpy() if hasattr(X_test, 'numpy') else X_test[:n_val]
                self.labels = y_test[:n_val].numpy() if hasattr(y_test, 'numpy') else y_test[:n_val]
            else:  # test
                X_test = dataset.X_test if dataset.X_test is not None else dataset.X_train
                y_test = dataset.y_test if dataset.y_test is not None else dataset.y_train
                n_val = len(X_test) // 2
                self.signals = X_test[n_val:].numpy() if hasattr(X_test, 'numpy') else X_test[n_val:]
                self.labels = y_test[n_val:].numpy() if hasattr(y_test, 'numpy') else y_test[n_val:]
            
            # 仅保留已知类与未知类
            allowed_classes = set(self.known_classes + self.unknown_classes)
            keep_mask = np.isin(self.labels, list(allowed_classes))
            self.signals = self.signals[keep_mask]
            self.labels = self.labels[keep_mask]

            # 将已知类映射为 0..K-1，未知类标记为 -1
            label_map = {cls: idx for idx, cls in enumerate(self.known_classes)}
            mapped_labels = np.full_like(self.labels, fill_value=-1, dtype=np.int64)
            for cls, idx in label_map.items():
                mapped_labels[self.labels == cls] = idx
            self.labels = mapped_labels

            # 标记未知类
            self.is_unknown = self.labels == -1
            
            # 对于训练集，移除未知类
            if self.split == 'train':
                known_mask = self.labels >= 0
                self.signals = self.signals[known_mask]
                self.labels = self.labels[known_mask]
                self.is_unknown = self.is_unknown[known_mask]
            
            print(f"CWRU {self.split} 集加载完成: {len(self)} 个样本")
            print(f"  已知类样本: {(~self.is_unknown).sum()}")
            print(f"  未知类样本: {self.is_unknown.sum()}")
            
        except Exception as e:
            print(f"加载 CWRU 数据失败: {e}")
            import traceback
            traceback.print_exc()
            # 使用模拟数据作为备选
            self._generate_dummy_data()
    
    def _generate_dummy_data(self) -> None:
        """生成模拟数据用于测试"""
        print("警告：使用模拟数据！")
        
        n_samples = 100 if self.split == 'train' else 30
        
        self.signals = np.random.randn(n_samples, self.signal_length).astype(np.float32)
        raw_labels = np.random.choice(
            self.known_classes + self.unknown_classes,
            size=n_samples
        )

        label_map = {cls: idx for idx, cls in enumerate(self.known_classes)}
        self.labels = np.full_like(raw_labels, fill_value=-1, dtype=np.int64)
        for cls, idx in label_map.items():
            self.labels[raw_labels == cls] = idx
        self.is_unknown = self.labels == -1
        
        # 训练集移除未知类
        if self.split == 'train':
            known_mask = ~self.is_unknown
            self.signals = self.signals[known_mask]
            self.labels = self.labels[known_mask]
            self.is_unknown = self.is_unknown[known_mask]
