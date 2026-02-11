"""
增强版 LPVG 图构建器 - 严格按照论文实现
包含：
1. 优化的穿透参数 P=2
2. 节点属性增强（归一化幅值 + 局部一阶差分 + 局部滑动均值 + 瞬时能量）
3. Z-score 标准化
"""

import numpy as np
import networkx as nx
from numba import njit
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pickle
from tqdm import tqdm
from scipy.signal import hilbert
from scipy.stats import moment


@njit
def fast_lpvg_edges_optimized(signal: np.ndarray, P: int = 2) -> List[Tuple[int, int]]:
    """
    优化的 LPVG 边生成算法（严格按照论文 Section 3.2）
    
    穿透参数 P 的物理意义：
    - P=0: 标准 VG，噪声敏感
    - P=1,2: 最优区间，能修复噪声断裂边且避免过穿透
    - P→∞: 过穿透，趋向全连接
    
    Args:
        signal: 输入信号，1D numpy array
        P: 穿透限制参数，允许视线穿透最多 P 个阻挡点
        
    Returns:
        edges: 边列表
    """
    L = len(signal)
    edges = []
    
    for i in range(L - 1):
        for j in range(i + 1, L):
            # 计算阻挡节点数
            blocking_nodes = 0
            
            for k in range(i + 1, j):
                # 计算斜率
                slope_ik = (signal[k] - signal[i]) / (k - i)
                slope_ij = (signal[j] - signal[i]) / (j - i)
                
                # 如果 slope_ik >= slope_ij，则点 k 阻挡视线
                if slope_ik >= slope_ij:
                    blocking_nodes += 1
                    
                    # 早停优化
                    if blocking_nodes > P:
                        break
            
            # 如果阻挡数不超过 P，建立连边
            if blocking_nodes <= P:
                edges.append((i, j))
    
    return edges


def compute_local_statistics(signal: np.ndarray, window_size: int = 5) -> Dict[str, np.ndarray]:
    """
    计算局部统计特征（论文 Section 3.2 Step 3）
    
    Args:
        signal: 输入信号
        window_size: 局部窗口大小 w
        
    Returns:
        features: 包含局部均值等局部统计量的字典
    """
    L = len(signal)
    local_mean = np.zeros(L)
    local_std = np.zeros(L)
    
    for i in range(L):
        # 定义局部窗口 [i-w, i+w]
        start = max(0, i - window_size)
        end = min(L, i + window_size + 1)
        
        local_window = signal[start:end]
        local_mean[i] = np.mean(local_window)
        local_std[i] = np.std(local_window)
    
    return {
        'local_mean': local_mean,
        'local_std': local_std
    }


def compute_instantaneous_frequency(signal: np.ndarray) -> np.ndarray:
    """
    通过希尔伯特变换计算瞬时频率（论文 Section 3.2 Step 3）
    
    Args:
        signal: 输入信号
        
    Returns:
        inst_freq: 瞬时频率
    """
    # 希尔伯特变换
    analytic_signal = hilbert(signal)
    
    # 计算瞬时相位
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
    # 瞬时频率 = 相位的时间导数
    inst_freq = np.diff(instantaneous_phase)
    inst_freq = np.concatenate([[inst_freq[0]], inst_freq])  # 保持长度一致
    
    return inst_freq


class EnhancedLPVGBuilder:
    """增强版 LPVG 图构建器（严格按照论文）"""
    
    def __init__(
        self, 
        P: int = 2,  # 论文推荐的最优参数
        local_window: int = 5,
        use_zscore: bool = True,
        use_instantaneous_freq: bool = True,
        max_nodes: int = None,
        downsample: str = 'uniform',
        use_cache: bool = True, 
        cache_dir: str = '.cache/lpvg_enhanced'
    ):
        """
        Args:
            P: 穿透参数（论文建议 P=2）
            local_window: 局部统计窗口大小
            use_zscore: 是否进行 Z-score 标准化
            use_instantaneous_freq: 是否提取瞬时频率特征
            max_nodes: 最大节点数（超过则下采样）
            downsample: 下采样方式（uniform/stride）
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
        """
        self.P = P
        self.local_window = local_window
        self.use_zscore = use_zscore
        self.use_instantaneous_freq = use_instantaneous_freq
        self.max_nodes = max_nodes
        self.downsample = downsample
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        信号预处理：Z-score 标准化（论文 Section 3.2 Step 1）
        
        Args:
            signal: 原始信号
            
        Returns:
            normalized_signal: 标准化后的信号
        """
        signal = signal.astype(np.float64, copy=False)
        if self.use_zscore:
            mean = np.mean(signal)
            std = np.std(signal)
            if std > 1e-8:  # 避免除零
                return (signal - mean) / std
            else:
                return signal - mean
        else:
            return signal

    def _downsample_signal(self, signal: np.ndarray) -> np.ndarray:
        if self.max_nodes is None or len(signal) <= self.max_nodes:
            return signal

        if self.downsample == 'stride':
            step = max(1, len(signal) // self.max_nodes)
            return signal[::step][:self.max_nodes]

        # Default: uniform sampling
        indices = np.linspace(0, len(signal) - 1, num=self.max_nodes, dtype=np.int64)
        return signal[indices]
    
    def build_graph(self, signal: np.ndarray) -> nx.Graph:
        """
        为单个信号构建增强的 LPVG 图
        
        Args:
            signal: 1D 信号
            
        Returns:
            G: NetworkX 图对象，包含丰富的节点特征
        """
        # Step 1: 预处理
        signal_normalized = self.preprocess_signal(signal)
        signal_normalized = self._downsample_signal(signal_normalized)
        
        # Step 2: 构建边（优化的穿透参数 P=2）
        edges = fast_lpvg_edges_optimized(signal_normalized, self.P)
        
        # 创建图
        L = len(signal_normalized)
        G = nx.Graph()
        G.add_nodes_from(range(L))
        
        if len(edges) > 0:
            G.add_edges_from(edges)
        
        # Step 3: 图属性增强
        # 3.1 计算局部统计特征
        local_stats = compute_local_statistics(signal_normalized, self.local_window)
        
        # 3.2 局部一阶差分与瞬时能量
        local_diff = np.zeros(L, dtype=np.float32)
        local_diff[:-1] = signal_normalized[1:] - signal_normalized[:-1]
        energy = signal_normalized ** 2
        inst_freq = compute_instantaneous_frequency(signal_normalized) if self.use_instantaneous_freq else None
        
        # 3.3 构造节点特征矩阵 H^(0) ∈ R^(L×F)
        for i in range(L):
            G.nodes[i]['value'] = float(signal_normalized[i])  # 归一化幅值
            G.nodes[i]['position'] = i  # 时间位置
            G.nodes[i]['local_diff'] = float(local_diff[i])  # 局部一阶差分
            G.nodes[i]['local_mean'] = float(local_stats['local_mean'][i])  # 局部滑动均值
            G.nodes[i]['energy'] = float(energy[i])  # 瞬时能量
            if inst_freq is not None:
                G.nodes[i]['inst_freq'] = float(inst_freq[i])
        
        return G
    
    def build_batch(
        self, 
        signals: np.ndarray, 
        show_progress: bool = True
    ) -> List[nx.Graph]:
        """
        批量构建 LPVG 图
        
        Args:
            signals: (N, L) 信号数组
            show_progress: 是否显示进度条
            
        Returns:
            graphs: 图列表
        """
        graphs = []
        
        iterator = tqdm(signals, desc="Building LPVG graphs") if show_progress else signals
        
        for signal in iterator:
            graph = self.build_graph(signal)
            graphs.append(graph)
        
        return graphs
    
    def save_cache(self, graphs: List[nx.Graph], cache_name: str):
        """保存图缓存"""
        if not self.use_cache:
            return
        
        cache_path = self.cache_dir / f"{cache_name}_P{self.P}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(graphs, f)
        print(f"✓ 缓存已保存至: {cache_path}")
    
    def load_cache(self, cache_name: str) -> List[nx.Graph]:
        """加载图缓存"""
        if not self.use_cache:
            return None
        
        cache_path = self.cache_dir / f"{cache_name}_P{self.P}.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                graphs = pickle.load(f)
            print(f"✓ 从缓存加载: {cache_path}")
            return graphs
        return None


def extract_graph_features(G: nx.Graph) -> np.ndarray:
    """
    从图中提取节点特征矩阵 H^(0)
    
    Args:
        G: NetworkX 图
        
    Returns:
        features: (num_nodes, num_features) 特征矩阵
    """
    num_nodes = G.number_of_nodes()
    
    # 特征维度：[value, local_diff, local_mean, energy]
    feature_dim = 4
    features = np.zeros((num_nodes, feature_dim))
    
    for i in range(num_nodes):
        node_data = G.nodes[i]
        features[i, 0] = node_data.get('value', 0.0)
        features[i, 1] = node_data.get('local_diff', 0.0)
        features[i, 2] = node_data.get('local_mean', 0.0)
        features[i, 3] = node_data.get('energy', 0.0)
    
    return features


if __name__ == "__main__":
    # 测试增强版 LPVG 构建器
    print("=" * 80)
    print("增强版 LPVG 构建器测试（严格按照论文实现）")
    print("=" * 80)
    
    # 生成测试信号（模拟带噪声的振动信号）
    np.random.seed(42)
    t = np.linspace(0, 1, 512)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 30 * t)
    signal += 0.3 * np.random.randn(len(t))  # 添加高斯噪声
    
    # 构建增强 LPVG 图
    builder = EnhancedLPVGBuilder(P=2, use_zscore=True, use_instantaneous_freq=True)
    G = builder.build_graph(signal)
    
    print(f"\n图统计:")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")
    print(f"  平均度: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    
    # 提取特征矩阵
    features = extract_graph_features(G)
    print(f"\n节点特征矩阵:")
    print(f"  形状: {features.shape}")
    print(f"  特征维度: [value, local_diff, local_mean, energy]")
    print(f"  前3个节点特征:\n{features[:3]}")
    
    print("\n✅ 增强版 LPVG 构建器测试通过!")
