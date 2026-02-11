"""数据预处理模块"""

import numpy as np
import torch
from scipy.fftpack import fft
from typing import Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, preprocessing_type: str = "fft", normalization: str = "zscore"):
        """
        初始化预处理器
        
        Args:
            preprocessing_type: 预处理类型 ("fft", "raw", "wavelet")
            normalization: 归一化方法 ("zscore", "minmax", "none")
        """
        self.preprocessing_type = preprocessing_type
        self.normalization = normalization
        self.scaler = None
        
    def preprocess(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        预处理数据
        
        Args:
            data: 输入数据
            
        Returns:
            预处理后的数据
        """
        # 转换为numpy数组
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        # 应用预处理
        if self.preprocessing_type == "fft":
            data = self._apply_fft(data)
        elif self.preprocessing_type == "raw":
            data = data
        elif self.preprocessing_type == "wavelet":
            data = self._apply_wavelet(data)
        else:
            raise ValueError(f"Unknown preprocessing type: {self.preprocessing_type}")
        
        # 应用归一化
        data = self._apply_normalization(data)
        
        # 转换回PyTorch张量
        return torch.tensor(data, dtype=torch.float32)
    
    def _apply_fft(self, data: np.ndarray) -> np.ndarray:
        """应用FFT变换"""
        # 确保数据是2D的 (如果是1D，添加一个维度)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # 计算FFT并取绝对值，保留前一半频谱
        fft_data = abs(fft(data))[:, :data.shape[1]//2]
        
        # 应用min-max变换
        fft_data = self._min_max_transform(fft_data)
        
        # 如果原始输入是1D，返回1D结果
        if fft_data.shape[0] == 1:
            fft_data = fft_data.flatten()
        
        return fft_data
    
    def _apply_wavelet(self, data: np.ndarray) -> np.ndarray:
        """应用小波变换"""
        # 这里可以实现小波变换
        # 目前返回原始数据
        return data
    
    def _min_max_transform(self, data: np.ndarray) -> np.ndarray:
        """Min-max变换"""
        # 确保数据是2D的
        if data.ndim == 1:
            data = data.reshape(1, -1)
            was_1d = True
        else:
            was_1d = False
        
        data_min = data.min(axis=1, keepdims=True)
        data = np.log(data - data_min + 1)
        
        # 如果原始输入是1D，返回1D结果
        if was_1d:
            data = data.flatten()
        
        return data
    
    def _apply_normalization(self, data: np.ndarray) -> np.ndarray:
        """应用归一化"""
        if self.normalization == "zscore":
            return self._zscore_normalize(data)
        elif self.normalization == "minmax":
            return self._minmax_normalize(data)
        elif self.normalization == "none":
            return data
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")
    
    def _zscore_normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score归一化"""
        # 确保数据是2D的
        if data.ndim == 1:
            data = data.reshape(1, -1)
            was_1d = True
        else:
            was_1d = False
        
        # 修正：使用真正的 Z-Score (x - mean) / std
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        epsilon = 1e-10
        normalized = (data - mean) / (std + epsilon)
        
        # 如果原始输入是1D，返回1D结果
        if was_1d:
            normalized = normalized.flatten()
        
        return normalized
    
    def _minmax_normalize(self, data: np.ndarray) -> np.ndarray:
        """Min-max归一化"""
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            return self.scaler.fit_transform(data)
        else:
            return self.scaler.transform(data)
    
    def fit_scaler(self, data: np.ndarray) -> None:
        """拟合归一化器"""
        if self.normalization == "minmax":
            self.scaler = MinMaxScaler()
            self.scaler.fit(data)
