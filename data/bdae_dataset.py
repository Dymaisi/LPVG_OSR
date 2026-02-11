"""BDAE航空发动机轴承数据集类 - LPVG_OSR项目独立版本"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

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
    
    def get_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        return {
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "train_size": len(self.X_train) if self.X_train is not None else 0,
            "test_size": len(self.X_test) if self.X_test is not None else 0,
            "labels": self.labels
        }


class BDAEWorkingCondition:
    """BDAE工况定义 - 27种LP-HP转速组合"""
    
    CONDITIONS = {
        1: (1000, 1200, "低速轻载-最小负荷"), 2: (1000, 1800, "低速轻载-中等负荷"), 3: (1000, 2400, "低速轻载-较大负荷"),
        4: (1500, 1800, "低速中载-低负荷"), 5: (1500, 2700, "低速中载-中等负荷"), 6: (1500, 3600, "低速中载-高负荷"),
        7: (2000, 2400, "中速变载-低负荷"), 8: (2000, 3600, "中速变载-中等负荷"), 9: (2000, 4800, "中速变载-高负荷"),
        10: (2500, 3000, "中速重载-标准负荷"), 11: (2500, 4500, "中速重载-高负荷"), 12: (2500, 6000, "中速重载-最大负荷"),
        13: (3000, 3600, "额定工况-标准运行"), 14: (3000, 5400, "额定工况-高负荷"),
        15: (3500, 4200, "高速工况-3500"), 16: (3600, 4320, "高速工况-3600"), 17: (3700, 4440, "高速工况-3700"),
        18: (3800, 4560, "高速工况-3800"), 19: (3900, 4680, "高速工况-3900"), 20: (4000, 4800, "高速工况-4000"),
        21: (4100, 4920, "高速工况-4100"), 22: (4200, 5040, "高速工况-4200"), 23: (4300, 5160, "高速工况-4300"),
        24: (4400, 5280, "高速工况-4400"), 25: (4500, 5400, "高速工况-4500"), 26: (4600, 5520, "高速工况-4600"),
        27: (5000, 6000, "高速工况-5000最大"),
    }
    
    @classmethod
    def get_condition(cls, condition_id: int) -> Tuple[int, int, str]:
        """获取工况信息"""
        return cls.CONDITIONS.get(condition_id, (None, None, "未知工况"))


class BDAEDataset(BaseDataset):
    """BDAE航空发动机轴承数据集
    
    类别标签定义:
    - Label 0: Normal (健康)
    - Label 1: Inner 0.5-0.5 (内圈故障 depth=0.5mm, length=0.5mm) -> data3
    - Label 2: Inner 0.5-1 (内圈故障 depth=0.5mm, length=1.0mm) -> data4
    - Label 3: Outer 0.5-0.5 (外圈故障 depth=0.5mm, length=0.5mm) -> data5
    """
    
    FAULT_TYPES = {
        0: "健康 (Normal)",
        1: "内圈故障0.5-0.5 (Inner 0.5mm-0.5mm)",
        2: "内圈故障0.5-1 (Inner 0.5mm-1.0mm)",
        3: "外圈故障0.5-0.5 (Outer 0.5mm-0.5mm)"
    }
    
    FILE_FAULT_MAP = {
        "data1.npy": 0, "data2.npy": 0,  # Normal
        "data3.npy": 1,  # Inner 0.5-0.5
        "data4.npy": 2,  # Inner 0.5-1
        "data5.npy": 3,  # Outer 0.5-0.5
    }
    
    def __init__(self, config: Dict[str, Any]):
        """初始化BDAE数据集"""
        super().__init__(config)
        
        # 数据集路径
        self.data_root = Path(config.get("data_root", "dataset/drive-download-20251103T072647Z-1-001"))
        
        # 工况配置
        self.working_conditions = config.get("working_conditions", [13, 14])
        self.is_source = config.get("is_source", True)
        
        # 类别配置
        if 'source_classes' in config or 'target_classes' in config:
            if self.is_source:
                self.allowed_classes = config.get("source_classes", [0, 1, 2])
            else:
                self.allowed_classes = config.get("target_classes", [0, 1, 2, 3])
            
            target_classes = config.get("target_classes", [0, 1, 2, 3])
            source_classes = config.get("source_classes", [0, 1, 2])
            
            self.known_classes = source_classes
            self.unknown_classes = [c for c in target_classes if c not in source_classes]
            self.include_unknown_in_target = True
        else:
            self.known_classes = config.get("known_classes", [0, 1])
            self.unknown_classes = config.get("unknown_classes", [2])
            self.include_unknown_in_target = config.get("include_unknown_in_target", True)
            
            if self.is_source:
                self.allowed_classes = self.known_classes
            else:
                self.allowed_classes = self.known_classes + (self.unknown_classes if self.include_unknown_in_target else [])
        
        # 数据处理参数
        self.train_ratio = config.get("train_ratio", 0.7)
        self.sequence_length = config.get("sequence_length", 2048)
        self.use_channels = config.get("use_channels", [0, 1, 2, 3, 4, 5])
        
        # 初始化预处理器
        self.preprocessor = DataPreprocessor(
            preprocessing_type=config.get("preprocessing", "fft"),
            normalization=config.get("normalization", "zscore")
        )
        
        # 标签映射
        self.label_map = self._build_label_map()
        self.class_labels = list(self.label_map.values())
        
        # 加载数据
        self.load_data()
        
    def _build_label_map(self) -> Dict[int, str]:
        """构建标签映射"""
        if self.is_source:
            return {i: self.FAULT_TYPES[cls] for i, cls in enumerate(self.known_classes)}
        else:
            label_map = {i: self.FAULT_TYPES[cls] for i, cls in enumerate(self.known_classes)}
            if self.include_unknown_in_target:
                label_map[len(self.known_classes)] = "未知故障"
            return label_map
    
    def _get_lp_speed_from_sample(self, sample: np.ndarray) -> int:
        """从样本中提取LP转速 (第7个通道索引6是转速信息)"""
        return int(sample[6, 0])
    
    def _match_working_condition(self, lp_speed: int) -> Optional[int]:
        """根据LP转速匹配工况ID"""
        for cid, (lp, hp, desc) in BDAEWorkingCondition.CONDITIONS.items():
            if lp == lp_speed and cid in self.working_conditions:
                return cid
        return None
    
    def load_data(self) -> None:
        """加载BDAE数据集"""
        print(f"\n{'='*60}")
        print(f"加载BDAE数据集 ({'源域' if self.is_source else '目标域'})")
        print(f"数据路径: {self.data_root}")
        print(f"{'='*60}")
        
        all_samples = []
        all_labels = []
        
        # 加载每个数据文件
        for filename, fault_label in self.FILE_FAULT_MAP.items():
            filepath = self.data_root / filename
            
            if not filepath.exists():
                print(f"  警告: {filename} 不存在，跳过")
                continue
            
            # 加载.npy文件
            data = np.load(filepath)  # shape: (N, 8, 20480)
            print(f"\n处理 {filename} (故障类型: {self.FAULT_TYPES[fault_label]})")
            
            # 根据域类型和类别配置筛选样本
            if self.is_source:
                if fault_label not in self.known_classes:
                    print(f"  跳过: 不是已知类")
                    continue
            else:
                if fault_label not in self.known_classes and fault_label not in self.unknown_classes:
                    print(f"  跳过: 不在配置的类别中")
                    continue
            
            # 筛选符合工况的样本
            condition_samples = []
            for i in range(data.shape[0]):
                sample = data[i]
                lp_speed = self._get_lp_speed_from_sample(sample)
                cid = self._match_working_condition(lp_speed)
                
                if cid is not None:
                    sample_channels = sample[self.use_channels, :]
                    condition_samples.append(sample_channels)
            
            if len(condition_samples) > 0:
                condition_samples = np.array(condition_samples)
                print(f"  筛选后样本数: {len(condition_samples)}")
                
                # 重映射标签
                if fault_label in self.known_classes:
                    mapped_label = self.known_classes.index(fault_label)
                else:
                    mapped_label = len(self.known_classes)  # 未知类
                
                all_samples.extend(condition_samples)
                all_labels.extend([mapped_label] * len(condition_samples))
        
        if len(all_samples) == 0:
            raise ValueError("没有加载到任何样本！请检查工况配置和数据路径。")
        
        # 转换为numpy数组
        all_samples = np.array(all_samples)
        all_labels = np.array(all_labels)
        
        print(f"\n数据加载完成: {len(all_samples)} 个样本")
        
        # 预处理数据
        print(f"开始预处理...")
        all_samples = self._preprocess_samples(all_samples)
        
        # 划分训练集和测试集
        self._split_train_test(all_samples, all_labels)
        
        # 更新数据集信息
        self.num_classes = len(self.label_map)
        self.feature_dim = self.X_train.shape[1]
        self.labels = self.class_labels
        
        print(f"{'='*60}\n")
    
    def _preprocess_samples(self, samples: np.ndarray) -> np.ndarray:
        """预处理样本数据"""
        processed_samples = []
        
        for sample in samples:
            # sample shape: (channels, 20480)
            channel_features = []
            for channel_data in sample:
                # 截取或填充到指定长度
                if len(channel_data) > self.sequence_length:
                    channel_data = channel_data[:self.sequence_length]
                elif len(channel_data) < self.sequence_length:
                    channel_data = np.pad(channel_data, 
                                         (0, self.sequence_length - len(channel_data)),
                                         mode='constant')
                
                # 应用预处理
                features = self.preprocessor.preprocess(channel_data)
                channel_features.append(features)
            
            # 合并所有通道的特征
            combined_features = np.concatenate(channel_features)
            processed_samples.append(combined_features)
        
        return np.array(processed_samples)
    
    def _split_train_test(self, X: np.ndarray, y: np.ndarray) -> None:
        """划分训练集和测试集"""
        unique_labels = np.unique(y)
        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []
        
        for label in unique_labels:
            indices = np.where(y == label)[0]
            np.random.shuffle(indices)
            
            split_idx = int(len(indices) * self.train_ratio)
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]
            
            X_train_list.append(X[train_indices])
            y_train_list.append(y[train_indices])
            X_test_list.append(X[test_indices])
            y_test_list.append(y[test_indices])
        
        # 合并并转换为Tensor
        self.X_train = torch.FloatTensor(np.vstack(X_train_list))
        self.y_train = torch.LongTensor(np.hstack(y_train_list))
        self.X_test = torch.FloatTensor(np.vstack(X_test_list))
        self.y_test = torch.LongTensor(np.hstack(y_test_list))
        
        print(f"  训练集: {len(self.X_train)} 样本")
        print(f"  测试集: {len(self.X_test)} 样本")
        print(f"  特征维度: {self.X_train.shape[1]}")
    
    def get_train_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取训练数据"""
        return self.X_train, self.y_train
    
    def get_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取测试数据"""
        return self.X_test, self.y_test
