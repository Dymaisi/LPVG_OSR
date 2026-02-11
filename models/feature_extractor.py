"""
图特征提取器
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List
import networkx as nx


class GraphFeatureExtractor(nn.Module):
    """从 LPVG 图提取特征"""
    
    def __init__(self, feature_list: List[str] = None):
        """
        Args:
            feature_list: 要提取的特征列表
        """
        super().__init__()
        
        if feature_list is None:
            # 默认特征
            self.feature_list = [
                'num_nodes', 'num_edges', 'avg_degree', 'max_degree',
                'std_degree', 'density', 'avg_clustering', 'transitivity',
                'assortativity'
            ]
        else:
            self.feature_list = feature_list
        
        self.feature_dim = len(self.feature_list)
    
    def extract(self, G: nx.Graph) -> np.ndarray:
        """提取图特征
        
        Args:
            G: NetworkX 图
            
        Returns:
            features: 1D 特征向量
        """
        features = []
        
        for feat_name in self.feature_list:
            if feat_name == 'num_nodes':
                features.append(G.number_of_nodes())
            elif feat_name == 'num_edges':
                features.append(G.number_of_edges())
            elif feat_name == 'avg_degree':
                degrees = [d for n, d in G.degree()]
                features.append(np.mean(degrees) if degrees else 0)
            elif feat_name == 'max_degree':
                degrees = [d for n, d in G.degree()]
                features.append(np.max(degrees) if degrees else 0)
            elif feat_name == 'std_degree':
                degrees = [d for n, d in G.degree()]
                features.append(np.std(degrees) if degrees else 0)
            elif feat_name == 'density':
                features.append(nx.density(G))
            elif feat_name == 'avg_clustering':
                clustering = list(nx.clustering(G).values())
                features.append(np.mean(clustering) if clustering else 0)
            elif feat_name == 'transitivity':
                features.append(nx.transitivity(G))
            elif feat_name == 'assortativity':
                try:
                    features.append(nx.degree_assortativity_coefficient(G))
                except:
                    features.append(0)
            else:
                features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def forward(self, graphs: List[nx.Graph]) -> torch.Tensor:
        """批量提取特征
        
        Args:
            graphs: 图列表
            
        Returns:
            features: (batch_size, feature_dim)
        """
        batch_features = []
        for G in graphs:
            feat = self.extract(G)
            batch_features.append(feat)
        
        return torch.FloatTensor(np.array(batch_features))
