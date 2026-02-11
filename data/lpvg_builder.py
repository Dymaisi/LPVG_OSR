"""
LPVG 图构建器
"""

import numpy as np
import networkx as nx
from numba import njit
from typing import List, Dict, Any
from pathlib import Path
import pickle
from tqdm import tqdm


@njit
def fast_lpvg_edges(signal: np.ndarray, m: int = 0) -> np.ndarray:
    """
    Numba 加速的 LPVG 边生成
    
    Args:
        signal: 输入信号，1D numpy array
        m: 穿透限制参数，允许视线穿透最多 m 个中间点
        
    Returns:
        edges: 边列表，shape (n_edges, 2)
    """
    n = len(signal)
    edges = []
    
    for i in range(n):
        for j in range(i+1, n):
            # 计算视线斜率
            slope_ij = (signal[j] - signal[i]) / (j - i)
            
            # 检查中间点是否阻挡视线
            penetration_count = 0
            visible = True
            
            for k in range(i+1, j):
                # 计算点 k 相对于直线 ij 的垂直偏移
                expected_value = signal[i] + slope_ij * (k - i)
                
                # 如果点 k 高于直线，说明阻挡了视线
                if signal[k] > expected_value:
                    penetration_count += 1
                    if penetration_count > m:
                        visible = False
                        break
            
            if visible:
                edges.append((i, j))
    
    return np.array(edges)


class LPVGBuilder:
    """LPVG 图构建器"""
    
    def __init__(self, m: int = 1, use_cache: bool = True, cache_dir: str = '.cache/lpvg'):
        """
        Args:
            m: 穿透参数
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
        """
        self.m = m
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def build_graph(self, signal: np.ndarray) -> nx.Graph:
        """为单个信号构建 LPVG 图
        
        Args:
            signal: 1D 信号
            
        Returns:
            G: NetworkX 图对象
        """
        # 构建边（使用 float64 以提高斜率精度）
        signal = signal.astype(np.float64, copy=False)
        edges = fast_lpvg_edges(signal, self.m)
        
        # 创建图
        G = nx.Graph()
        G.add_nodes_from(range(len(signal)))
        
        if len(edges) > 0:
            G.add_edges_from(edges)
        
        # 添加节点属性（信号值）
        for i, value in enumerate(signal):
            G.nodes[i]['value'] = float(value)
            G.nodes[i]['position'] = i
        
        return G
    
    def build_batch(self, signals: np.ndarray, show_progress: bool = True) -> List[nx.Graph]:
        """批量构建 LPVG 图
        
        Args:
            signals: (N, L) 信号数组
            show_progress: 是否显示进度条
            
        Returns:
            graphs: 图列表
        """
        graphs = []
        
        iterator = tqdm(signals, desc="构建 LPVG 图") if show_progress else signals
        
        for signal in iterator:
            G = self.build_graph(signal)
            graphs.append(G)
        
        return graphs
    
    def save_cache(self, graphs: List[nx.Graph], cache_key: str) -> None:
        """保存图缓存
        
        Args:
            graphs: 图列表
            cache_key: 缓存键
        """
        if not self.use_cache:
            return
            
        cache_file = self.cache_dir / f"{cache_key}_m{self.m}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(graphs, f)
    
    def load_cache(self, cache_key: str) -> List[nx.Graph]:
        """加载图缓存
        
        Args:
            cache_key: 缓存键
            
        Returns:
            graphs: 图列表，如果不存在返回 None
        """
        if not self.use_cache:
            return None
            
        cache_file = self.cache_dir / f"{cache_key}_m{self.m}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None


def extract_graph_features(G: nx.Graph) -> np.ndarray:
    """提取图的拓扑特征
    
    Args:
        G: NetworkX 图对象
        
    Returns:
        features: 1D 特征向量
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    # 基本特征
    features = {
        'num_nodes': n_nodes,
        'num_edges': n_edges,
    }
    
    # 度相关特征
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    
    features['avg_degree'] = np.mean(degree_values) if degree_values else 0
    features['max_degree'] = np.max(degree_values) if degree_values else 0
    features['std_degree'] = np.std(degree_values) if degree_values else 0
    
    # 密度
    features['density'] = nx.density(G)
    
    # 大图跳过高复杂度统计，避免长时间阻塞
    large_graph = (n_nodes > 512) or (n_edges > 20000)
    if not large_graph:
        # 聚类系数
        clustering = nx.clustering(G)
        clustering_values = list(clustering.values())
        features['avg_clustering'] = np.mean(clustering_values) if clustering_values else 0
        
        # 传递性
        features['transitivity'] = nx.transitivity(G)
        
        # 度相关性（assortativity）
        try:
            features['assortativity'] = nx.degree_assortativity_coefficient(G)
        except:
            features['assortativity'] = 0
        
        # 连通性
        if nx.is_connected(G):
            features['diameter'] = nx.diameter(G)
            features['avg_shortest_path'] = nx.average_shortest_path_length(G)
        else:
            features['diameter'] = 0
            features['avg_shortest_path'] = 0
    else:
        features['avg_clustering'] = 0
        features['transitivity'] = 0
        features['assortativity'] = 0
        features['diameter'] = 0
        features['avg_shortest_path'] = 0
    
    # 转换为向量
    feature_vector = np.array([
        features['num_nodes'],
        features['num_edges'],
        features['avg_degree'],
        features['max_degree'],
        features['std_degree'],
        features['density'],
        features['avg_clustering'],
        features['transitivity'],
        features['assortativity'],
        features['diameter'],
        features['avg_shortest_path']
    ], dtype=np.float32)
    
    return feature_vector
