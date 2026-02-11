"""
BDAE 开放集数据集
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any
from .base_dataset import BaseOSRDataset


class BDAEOpenSet(BaseOSRDataset):
    """BDAE 航空发动机轴承数据集 - 开放集版本
    
    类别定义：
    - 0: Normal (健康)
    - 1: Inner 0.5-0.5 (内圈故障 depth=0.5mm, length=0.5mm)
    - 2: Inner 0.5-1.0 (内圈故障 depth=0.5mm, length=1.0mm)
    - 3: Outer 0.5-0.5 (外圈故障 depth=0.5mm, length=0.5mm)
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
        self.channel_index = config.get('channel_index', 0)
        self.working_conditions = config.get('working_conditions', [1, 2, 3])
        
        # 加载数据
        self.load_data()
        
    def load_data(self) -> None:
        """加载 BDAE 数据"""
        try:
            # 直接从 .npy 文件加载
            self._load_from_npy()
            
            print(f"BDAE {self.split} 集加载完成: {len(self)} 个样本")
            print(f"  已知类样本: {(~self.is_unknown).sum()}")
            print(f"  未知类样本: {self.is_unknown.sum()}")
            
        except Exception as e:
            print(f"加载 BDAE 数据失败: {e}")
            import traceback
            traceback.print_exc()
            # 使用模拟数据作为备选
            self._generate_dummy_data()
    
    def _load_from_npy(self) -> None:
        """从 .npy 文件加载数据"""
        # 文件映射
        fault_files = {
            "data1.npy": 0,  # Normal
            "data2.npy": 0,  # Normal
            "data3.npy": 1,  # Inner 0.5-0.5
            "data4.npy": 2,  # Inner 0.5-1.0
            "data5.npy": 3,  # Outer 0.5-0.5
        }
        
        all_signals = []
        all_labels = []
        
        for filename, label in fault_files.items():
            filepath = self.data_root / filename
            if not filepath.exists():
                print(f"警告：文件不存在 {filename}")
                continue
            
            # 加载数据
            data = np.load(filepath)  # (N, 8, 20480)
            
            # 提取指定通道
            channel_data = data[:, self.channel_index, :]  # (N, 20480)
            
            # 切分为固定长度的片段
            signals = self._segment_signals(channel_data)
            
            # 添加到列表
            all_signals.extend(signals)
            all_labels.extend([label] * len(signals))
        
        # 转换为数组
        all_signals = np.array(all_signals, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)
        
        # 划分数据集
        train_ratio = self.config.get('train_ratio', 0.7)
        val_ratio = self.config.get('val_ratio', 0.15)
        
        n_total = len(all_signals)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # 随机打乱
        indices = np.random.permutation(n_total)
        
        if self.split == 'train':
            idx = indices[:n_train]
        elif self.split == 'val':
            idx = indices[n_train:n_train+n_val]
        else:  # test
            idx = indices[n_train+n_val:]
        
        self.signals = all_signals[idx]
        self.labels = all_labels[idx]
        
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
        
        # 归一化
        if self.config.get('preprocessing', {}).get('normalization') == 'zscore':
            self.signals = (self.signals - self.signals.mean(axis=1, keepdims=True)) / \
                          (self.signals.std(axis=1, keepdims=True) + 1e-8)
    
    def _segment_signals(self, signals: np.ndarray) -> list:
        """将长信号切分为固定长度的片段
        
        Args:
            signals: (N, L) 原始信号
            
        Returns:
            segments: 切分后的片段列表
        """
        segments = []
        
        for signal in signals:
            # 计算步长
            step = int(self.signal_length * (1 - self.overlap))
            
            # 滑动窗口切分
            for start in range(0, len(signal) - self.signal_length + 1, step):
                segment = signal[start:start + self.signal_length]
                segments.append(segment)
        
        return segments
    
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
            known_mask = self.labels >= 0
            self.signals = self.signals[known_mask]
            self.labels = self.labels[known_mask]
            self.is_unknown = self.is_unknown[known_mask]
