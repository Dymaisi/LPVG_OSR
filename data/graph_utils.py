"""
图数据转换工具 - 将 NetworkX 图转换为 PyTorch 张量
"""

import torch
import numpy as np
import networkx as nx
from typing import List, Tuple


def _infer_feature_dim(graphs: List[nx.Graph]) -> int:
    for G in graphs:
        if G.number_of_nodes() == 0:
            continue
        node_idx = next(iter(G.nodes))
        node_data = G.nodes[node_idx]
        if 'inst_freq' in node_data:
            return 5
        if any(key in node_data for key in ('local_diff', 'local_mean', 'energy')):
            return 4
    return 1


def networkx_to_sparse(
    graphs: List[nx.Graph],
    add_self_loops: bool = True
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """将 NetworkX 图转换为稀疏表示
    
    Args:
        graphs: NetworkX 图列表
        
    Returns:
        node_features_list: 每个图的节点特征张量
        edge_index_list: 每个图的边索引 (2, E)
    """
    feature_dim = _infer_feature_dim(graphs)
    
    # 初始化张量
    node_features_list = []
    edge_index_list = []
    
    for G in graphs:
        num_nodes = G.number_of_nodes()
        
        # 提取节点特征 (value + 可选增强特征)
        node_feat = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        for i in range(num_nodes):
            node_data = G.nodes[i]
            node_feat[i, 0] = node_data.get('value', 0.0)
            if feature_dim == 4:
                node_feat[i, 1] = node_data.get('local_diff', 0.0)
                node_feat[i, 2] = node_data.get('local_mean', 0.0)
                node_feat[i, 3] = node_data.get('energy', 0.0)
            elif feature_dim == 5:
                node_feat[i, 1] = node_data.get('local_diff', 0.0)
                node_feat[i, 2] = node_data.get('local_mean', 0.0)
                node_feat[i, 3] = node_data.get('energy', 0.0)
                node_feat[i, 4] = node_data.get('inst_freq', 0.0)
        
        node_features_list.append(torch.FloatTensor(node_feat))

        # 构建边索引 (无向图需要双向边)
        if G.number_of_edges() > 0:
            edges = np.array(list(G.edges()), dtype=np.int64)
            src = edges[:, 0]
            dst = edges[:, 1]
            src = np.concatenate([src, dst])
            dst = np.concatenate([dst, edges[:, 0]])
        else:
            src = np.array([], dtype=np.int64)
            dst = np.array([], dtype=np.int64)

        if add_self_loops:
            self_loops = np.arange(num_nodes, dtype=np.int64)
            src = np.concatenate([src, self_loops])
            dst = np.concatenate([dst, self_loops])

        edge_index = torch.LongTensor(np.stack([src, dst], axis=0)) if src.size else torch.LongTensor(np.zeros((2, 0), dtype=np.int64))
        edge_index_list.append(edge_index)
    
    return node_features_list, edge_index_list


class GraphBatchCollator:
    """图数据批次整理器"""
    
    def __init__(self, max_nodes: int = None, add_self_loops: bool = True):
        """
        Args:
            max_nodes: 最大节点数，如果为 None 则动态确定
        """
        self.max_nodes = max_nodes
        self.add_self_loops = add_self_loops
    
    def __call__(self, batch: List[Tuple]) -> Tuple:
        """整理批次数据
        
        Args:
            batch: [(graph, label, is_unknown), ...]
            
        Returns:
            node_features, adj_matrices, labels, is_unknown
        """
        graphs, labels, is_unknown = zip(*batch)
        
        # 转换图为稀疏表示
        node_features, edge_index = networkx_to_sparse(
            list(graphs),
            add_self_loops=self.add_self_loops
        )
        
        # 转换标签
        labels = torch.LongTensor(labels)
        is_unknown = torch.FloatTensor(is_unknown)
        
        return node_features, edge_index, labels, is_unknown
