"""CWRU数据集类 - LPVG_OSR项目独立版本"""

import contextlib
import io
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
try:
    from cwru import CWRU
except ImportError:
    print("警告：未安装cwru包，请运行 pip install cwru")
    CWRU = None

# 本地导入（绝对导入）
try:
    from preprocessing import DataPreprocessor
except ImportError:
    from data.preprocessing import DataPreprocessor


class BaseDataset:
    """基础数据集类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化数据集"""
        self.config = config
        self.X_train: torch.Tensor = None
        self.y_train: torch.Tensor = None
        self.X_test: torch.Tensor = None
        self.y_test: torch.Tensor = None
        self.num_classes: int = 0
        self.feature_dim: int = 0
        self.labels: list = None


class CWRUDataset(BaseDataset):
    """CWRU数据集类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化CWRU数据集
        
        Args:
            config: 数据集配置
        """
        super().__init__(config)
        
        self.fault_type = config.get("fault_type", "12DriveEndFault")
        self.rpm = config.get("rpm", "1797")
        self.sequence_length = config.get("sequence_length", 1024)
        self.retain_labels = config.get("retain_labels", None)
        
        # 开放集相关参数
        self.is_target = config.get("is_target", False)
        self.source_retain_labels = config.get("source_retain_labels", None)
        
        # 初始化预处理器
        self.preprocessor = DataPreprocessor(
            preprocessing_type=config.get("preprocessing", "fft"),
            normalization=config.get("normalization", "zscore")
        )
        
        # 加载数据
        self.load_data()
        
    def load_data(self) -> None:
        """加载CWRU数据"""
        if CWRU is None:
            raise ImportError("未安装cwru包，请运行: pip install cwru")
            
        # 抑制CWRU包的打印输出
        with contextlib.redirect_stdout(io.StringIO()):
            # 使用原始CWRU数据加载器
            cwru_data = CWRU(self.fault_type, self.rpm, self.sequence_length)
        
        # 获取原始数据
        X_raw = torch.tensor(cwru_data.X_train, dtype=torch.float32)
        y_raw = torch.tensor(cwru_data.y_train, dtype=torch.long)
        
        # 调试：打印真正原始的CWRU标签（在任何处理之前）
        print(f"=== CWRU原始数据（{self.fault_type}, {self.rpm}）===")
        print(f"原始标签: {sorted(torch.unique(y_raw).tolist())}")
        labels_raw, counts_raw = torch.unique(y_raw, return_counts=True)
        print("原始标签统计:")
        for label, count in zip(labels_raw, counts_raw):
            print(f"  标签 {label.item()}: {count.item()} 个样本")
        print("=== 原始数据结束 ===")
        
        # 预处理数据
        X_processed = self.preprocessor.preprocess(X_raw)
        
        # 调试：打印预处理后的数据统计
        print(f"预处理后数据统计: Mean={X_processed.mean():.4f}, Std={X_processed.std():.4f}, Min={X_processed.min():.4f}, Max={X_processed.max():.4f}")
        if torch.isnan(X_processed).any():
            print("警告：预处理后数据包含 NaN!")
        
        X_processed, y_processed = self._filter_labels(X_processed, y_raw)
        
        # 设置数据
        self.X_train = X_processed
        self.y_train = y_processed
        
        # 设置数据集信息
        self.num_classes = len(torch.unique(y_processed))
        self.feature_dim = X_processed.shape[1]
        self.labels = cwru_data.labels
        
    def _filter_labels(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据retain_labels过滤数据并实现开放集映射
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            过滤后的特征和标签
        """
        if self.retain_labels is None:
            # 如果没有指定retain_labels，则返回所有数据
            print(f"retain_labels为None，保留所有数据")
            return X, y
            
        print(f"{'目标域' if self.is_target else '源域'} retain_labels: {self.retain_labels}")
        
        if self.is_target and self.source_retain_labels is not None:
            # 目标域：实现开放集映射
            return self._create_open_set_mapping(X, y)
        else:
            # 源域：简单的连续映射
            return self._create_source_mapping(X, y)
    
    def _create_source_mapping(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """为源域创建连续标签映射"""
        # 创建标签映射：retain_labels_source: [0, 1, 5, 6, 8, 9, 15] -> [0,1,2,3,4,5,6]
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(self.retain_labels)}
        print(f"源域标签映射: {label_mapping}")
        
        # 过滤数据
        retain_indices = [i for i, label in enumerate(y) if label.item() in self.retain_labels]
        X_filtered = X[retain_indices]
        y_filtered = torch.tensor([label_mapping[label.item()] for label in y[retain_indices]])
        
        print(f"源域过滤前标签: {sorted(torch.unique(y).tolist())}")
        print(f"源域过滤后标签: {sorted(torch.unique(y_filtered).tolist())}")
        
        return X_filtered, y_filtered
    
    def _create_open_set_mapping(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """为目标域创建开放集映射"""
        source_label_mapping = {old_label: new_label for new_label, old_label in enumerate(self.source_retain_labels)}
        unknown_label = len(self.source_retain_labels)  # 未知类标签
        
        target_label_mapping = {}
        for old_label in self.retain_labels:
            if old_label in self.source_retain_labels:
                # 已知类：映射到源域的连续编号
                target_label_mapping[old_label] = source_label_mapping[old_label]
            else:
                # 未知类：映射到unknown_label
                target_label_mapping[old_label] = unknown_label
        
        print(f"源域标签映射: {source_label_mapping}")
        print(f"目标域标签映射: {target_label_mapping}")
        
        # 过滤数据
        retain_indices = [i for i, label in enumerate(y) if label.item() in self.retain_labels]
        X_filtered = X[retain_indices]
        y_filtered = torch.tensor([target_label_mapping[label.item()] for label in y[retain_indices]])
        
        print(f"目标域过滤前标签: {sorted(torch.unique(y).tolist())}")
        print(f"目标域过滤后标签: {sorted(torch.unique(y_filtered).tolist())}")
        
        return X_filtered, y_filtered
    
    def get_train_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取训练数据"""
        return self.X_train, self.y_train
    
    def get_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取测试数据"""
        return self.X_test, self.y_test
