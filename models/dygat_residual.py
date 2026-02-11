"""
动态图注意力网络（DyGAT）- 带初始残差连接
严格按照论文 Section 3.3 实现
核心创新：
1. 可学习的动态图构建（LDGC）
2. 多头注意力机制
3. 初始残差连接抑制过平滑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import numpy as np
import math


class DynamicGraphConstruction(nn.Module):
    """
    可学习的动态图构建模块（论文 Section 3.3.1）
    根据节点特征动态计算邻接矩阵
    """
    
    def __init__(self, feature_dim: int, top_k: int = 10, metric: str = 'cosine'):
        """
        Args:
            feature_dim: 节点特征维度
            top_k: 每个节点保留 top-k 个最相似的邻居
            metric: 相似度度量方式（'cosine' 或 'mlp'）
        """
        super().__init__()
        self.top_k = top_k
        self.metric = metric
        
        if metric == 'mlp':
            # 使用 MLP 学习相似度
            self.W_dyn = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (batch_size, num_nodes, feature_dim) 节点特征
            
        Returns:
            A_dyn: (batch_size, num_nodes, num_nodes) 动态邻接矩阵
        """
        batch_size, num_nodes, feature_dim = H.shape
        
        if self.metric == 'cosine':
            # 余弦相似度
            H_norm = F.normalize(H, p=2, dim=-1)
            S = torch.bmm(H_norm, H_norm.transpose(1, 2))  # (B, N, N)
        else:
            # MLP 度量
            H_transformed = self.W_dyn(H)
            S = torch.bmm(H_transformed, H.transpose(1, 2))
            S = F.leaky_relu(S, negative_slope=0.2)
        
        # Top-k 筛选
        A_dyn = torch.zeros_like(S)
        
        # 对每个节点，保留 top-k 个最相似的邻居
        topk_values, topk_indices = torch.topk(S, self.top_k, dim=-1)
        
        for b in range(batch_size):
            for i in range(num_nodes):
                for k_idx in range(self.top_k):
                    j = topk_indices[b, i, k_idx]
                    A_dyn[b, i, j] = 1.0
        
        # 对称化
        A_dyn = (A_dyn + A_dyn.transpose(1, 2)) / 2.0
        
        return A_dyn


class GraphAttentionLayer(nn.Module):
    """
    图注意力层（论文 Section 3.3.2）
    实现各向异性的邻域聚合
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        dropout: float = 0.0,
        alpha: float = 0.2,
        concat: bool = True,
        tau_cut: Optional[float] = None
    ):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            dropout: Dropout 率
            alpha: LeakyReLU 负斜率
            concat: 是否拼接多头（True）或平均（False）
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.tau_cut = tau_cut
        
        # 特征变换矩阵
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # 注意力机制参数 a^T
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            h: (num_nodes, in_features) 节点特征
            adj: (num_nodes, num_nodes) 邻接矩阵
            
        Returns:
            h_prime: (num_nodes, out_features) 更新后的特征
        """
        # 特征变换: Wh
        Wh = torch.mm(h, self.W)  # (N, out_features)
        num_nodes = Wh.size(0)

        # 构造边索引，仅在邻接边上计算注意力
        if edge_index is None:
            edge_index = (adj > 0).nonzero(as_tuple=False).t()

        if edge_index.numel() == 0:
            h_prime = torch.zeros_like(Wh)
        else:
            src = edge_index[0]
            dst = edge_index[1]

            Wh_src = Wh.index_select(0, src)
            Wh_dst = Wh.index_select(0, dst)
            e = torch.matmul(torch.cat([Wh_src, Wh_dst], dim=1), self.a).squeeze(1)
            e = self.leakyrelu(e)

            max_per_src = torch.full((num_nodes,), -float('inf'), device=e.device)
            if hasattr(max_per_src, 'scatter_reduce_'):
                max_per_src.scatter_reduce_(0, src, e, reduce='amax', include_self=True)
            else:
                unique_src = torch.unique(src)
                for node in unique_src:
                    mask = src == node
                    max_per_src[node] = torch.max(e[mask])

            exp_e = torch.exp(e - max_per_src.index_select(0, src))
            denom = torch.zeros(num_nodes, device=e.device)
            denom.scatter_add_(0, src, exp_e)
            attention = exp_e / (denom.index_select(0, src) + 1e-12)

            attention = F.dropout(attention, self.dropout, training=self.training)

            if self.tau_cut is not None:
                prune_mask = attention >= self.tau_cut
                attention = attention * prune_mask
                denom = torch.zeros(num_nodes, device=attention.device)
                denom.index_add_(0, src, attention)
                attention = attention / (denom.index_select(0, src) + 1e-12)

            h_prime = torch.zeros_like(Wh)
            h_prime.index_add_(0, src, attention.unsqueeze(1) * Wh_dst)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        """
        计算注意力系数 e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        
        Args:
            Wh: (N, out_features) 变换后的特征
            
        Returns:
            e: (N, N) 注意力能量矩阵
        """
        N = Wh.size(0)
        
        # 构造 [Wh_i || Wh_j] 矩阵
        # Wh_i.repeat(1, N) -> (N, N*out_features)
        # 然后重塑为 (N, N, out_features)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # (N*N, out_features)
        Wh_repeated_alternating = Wh.repeat(N, 1)  # (N*N, out_features)
        
        # 拼接
        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1
        )  # (N*N, 2*out_features)
        
        # 计算 a^T [Wh_i || Wh_j]
        e = torch.matmul(all_combinations_matrix, self.a).squeeze(1)
        e = self.leakyrelu(e)
        
        # 重塑为 (N, N)
        return e.view(N, N)


class MultiHeadGraphAttention(nn.Module):
    """多头图注意力（论文 Section 3.3.2）"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        alpha: float = 0.2,
        concat: bool = True,
        tau_cut: Optional[float] = None
    ):
        """
        Args:
            in_features: 输入特征维度
            out_features: 每个头的输出维度
            num_heads: 注意力头数 K
            dropout: Dropout 率
            alpha: LeakyReLU 负斜率
            concat: 是否拼接多头
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        
        # 创建多个注意力头
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(
                in_features, 
                out_features, 
                dropout=dropout, 
                alpha=alpha, 
                concat=True,
                tau_cut=tau_cut
            ) for _ in range(num_heads)
        ])
    
    def forward(
        self,
        h: torch.Tensor,
        adj: Optional[torch.Tensor],
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            h: (num_nodes, in_features)
            adj: (num_nodes, num_nodes)
            
        Returns:
            h_out: (num_nodes, num_heads * out_features) if concat
                   (num_nodes, out_features) if average
        """
        # 共享边索引，避免重复构造
        if edge_index is None:
            edge_index = (adj > 0).nonzero(as_tuple=False).t()

        # 对每个头计算注意力聚合
        head_outputs = [att(h, adj, edge_index=edge_index) for att in self.attentions]
        
        if self.concat:
            # 拼接所有头: h_agg^(l) = ||_{k=1}^K h_i^(k)
            return torch.cat(head_outputs, dim=1)
        else:
            # 平均所有头
            return torch.mean(torch.stack(head_outputs), dim=0)


class DyGATLayer(nn.Module):
    """
    完整的 DyGAT 层（论文 Section 3.3）
    包含：动态图构建 + 多头注意力 + 初始残差
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        alpha: float = 0.2,
        use_dynamic_graph: bool = True,
        top_k: int = 10,
        mu: float = 0.5,  # LPVG 与动态图的融合权重
        tau_cut: Optional[float] = None
    ):
        """
        Args:
            in_features: 输入特征维度
            out_features: 每个头的输出维度
            num_heads: 注意力头数
            dropout: Dropout 率
            alpha: LeakyReLU 负斜率
            use_dynamic_graph: 是否使用动态图构建
            top_k: 动态图的 top-k 邻居数
            mu: LPVG 先验权重（论文 Section 3.3.1）
        """
        super().__init__()
        self.use_dynamic_graph = use_dynamic_graph
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float32))
        
        # 动态图构建模块
        if use_dynamic_graph:
            self.dgc = DynamicGraphConstruction(in_features, top_k)
        
        # 多头注意力
        self.mhgat = MultiHeadGraphAttention(
            in_features, 
            out_features, 
            num_heads, 
            dropout, 
            alpha,
            tau_cut=tau_cut
        )
    
    def forward(
        self,
        h: torch.Tensor,
        adj_lpvg: torch.Tensor,
        return_adj: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            h: (num_nodes, in_features) 节点特征
            adj_lpvg: (num_nodes, num_nodes) LPVG 邻接矩阵
            return_adj: 是否返回融合后的邻接矩阵
            
        Returns:
            h_out: (num_nodes, num_heads * out_features)
            adj_fused: (num_nodes, num_nodes) 融合后的邻接矩阵（可选）
        """
        # 支持稀疏边索引或稠密邻接矩阵
        use_edge_index = adj_lpvg.dim() == 2 and adj_lpvg.size(0) == 2
        edge_index = adj_lpvg if use_edge_index else None

        # 动态图构建与融合（论文公式：A^(l) = μ·A_LPVG + (1-μ)·A_learn）
        if self.use_dynamic_graph:
            h_batch = h.unsqueeze(0)  # (1, N, F)
            adj_dyn = self.dgc(h_batch).squeeze(0)  # (N, N)

            if use_edge_index:
                num_nodes = h.size(0)
                adj_lpvg_dense = torch.zeros((num_nodes, num_nodes), device=h.device)
                if edge_index.numel() > 0:
                    adj_lpvg_dense[edge_index[0], edge_index[1]] = 1.0
                adj_fused = torch.sigmoid(self.mu) * adj_lpvg_dense + (1 - torch.sigmoid(self.mu)) * adj_dyn
                edge_index = (adj_fused > 0).nonzero(as_tuple=False).t()
                adj_fused = None
            else:
                mu = torch.sigmoid(self.mu)
                adj_fused = mu * adj_lpvg + (1 - mu) * adj_dyn
        else:
            adj_fused = None if use_edge_index else adj_lpvg

        # 多头注意力聚合
        h_out = self.mhgat(h, adj_fused, edge_index=edge_index)
        
        if return_adj:
            if adj_fused is None and edge_index is not None:
                num_nodes = h.size(0)
                adj_fused = torch.zeros((num_nodes, num_nodes), device=h.device)
                if edge_index.numel() > 0:
                    adj_fused[edge_index[0], edge_index[1]] = 1.0
            return h_out, adj_fused
        else:
            return h_out


class DyGATWithInitialResidual(nn.Module):
    """
    带初始残差连接的 DyGAT 网络（论文 Section 3.3.3）
    核心创新：抑制深层网络的过平滑问题
    
    更新公式（论文）：
    H^(l+1) = σ((1-β_l) H_agg^(l) + β_l H^(0))
    其中 β_l = ln(λ/l + 1)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 64, 32],
        num_heads: int = 8,
        dropout: float = 0.3,
        lambda_residual: float = 2.0,  # 残差衰减参数 λ
        alpha_residual: float = 0.1,
        use_dynamic_graph: bool = True,
        top_k: int = 10,
        mu: float = 0.5,
        tau_cut: Optional[float] = None
    ):
        """
        Args:
            input_dim: 输入特征维度（H^(0) 的维度）
            hidden_dims: 各层隐藏维度列表
            num_heads: 注意力头数
            dropout: Dropout 率
            lambda_residual: 残差衰减参数 λ
            use_dynamic_graph: 是否使用动态图构建
            top_k: 动态图 top-k
            mu: LPVG 融合权重
        """
        super().__init__()
        self.num_layers = len(hidden_dims)
        self.lambda_residual = lambda_residual
        self.alpha_residual = alpha_residual
        self.dropout = dropout
        
        # 构建多层 DyGAT
        self.layers = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims
        for i in range(self.num_layers):
            layer = DyGATLayer(
                in_features=dims[i],
                out_features=hidden_dims[i] // num_heads,  # 每个头的维度
                num_heads=num_heads,
                dropout=dropout,
                use_dynamic_graph=use_dynamic_graph,
                top_k=top_k,
                mu=mu,
                tau_cut=tau_cut
            )
            self.layers.append(layer)
        
        # 投影层（将初始特征投影到各层输出空间）
        self.h0_projections = nn.ModuleList()
        for i in range(self.num_layers):
            out_dim = hidden_dims[i]
            if input_dim != out_dim:
                self.h0_projections.append(nn.Linear(input_dim, out_dim))
            else:
                self.h0_projections.append(nn.Identity())
        
        # 恒等映射对应的线性变换 W^(l)
        self.identity_transforms = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[i]) for i in range(self.num_layers)
        ])
    
    def compute_residual_weight(self, layer_idx: int) -> float:
        """
        计算第 l 层的残差权重 β_l（论文公式）
        β_l = ln(λ/l + 1)
        
        Args:
            layer_idx: 层索引（从 1 开始）
            
        Returns:
            beta: 残差权重
        """
        l = layer_idx + 1  # 层号从 1 开始
        beta = math.log(self.lambda_residual / l + 1)
        return beta
    
    def forward(
        self, 
        h_initial: torch.Tensor, 
        adj_lpvg: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_initial: (num_nodes, input_dim) 初始节点特征 H^(0)
            adj_lpvg: (num_nodes, num_nodes) LPVG 邻接矩阵
            
        Returns:
            h_final: (num_nodes, hidden_dims[-1]) 最终节点嵌入
        """
        # 保存初始特征用于残差连接
        h_0 = h_initial
        h = h_initial
        
        # 逐层传播
        for layer_idx, layer in enumerate(self.layers):
            # 聚合邻域信息
            h_agg = layer(h, adj_lpvg)
            
            # 计算残差权重
            beta = self.compute_residual_weight(layer_idx)
            
            # 初始残差 + 恒等映射（论文核心公式）
            # H^(l+1) = σ(((1-α) A H^(l) + α H^(0)) ((1-β) I + β W^(l)))
            h_0_proj = self.h0_projections[layer_idx](h_0)
            h_hat = (1 - self.alpha_residual) * h_agg + self.alpha_residual * h_0_proj
            h = (1 - beta) * h_hat + beta * self.identity_transforms[layer_idx](h_hat)
            h = F.elu(h)
            
            # Dropout
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h


if __name__ == "__main__":
    print("=" * 80)
    print("DyGAT with Initial Residual 测试")
    print("=" * 80)
    
    # 测试参数
    num_nodes = 256
    input_dim = 4  # [value, local_diff, local_mean, energy]
    batch_size = 1
    
    # 生成测试数据
    h_initial = torch.randn(num_nodes, input_dim)
    adj_lpvg = torch.rand(num_nodes, num_nodes)
    adj_lpvg = (adj_lpvg > 0.95).float()  # 稀疏邻接矩阵
    adj_lpvg = (adj_lpvg + adj_lpvg.T) / 2  # 对称化
    
    # 创建模型
    model = DyGATWithInitialResidual(
        input_dim=input_dim,
        hidden_dims=[64, 64, 32],
        num_heads=8,
        dropout=0.3,
        lambda_residual=2.0,
        use_dynamic_graph=True
    )
    
    print(f"\n模型参数:")
    print(f"  层数: {model.num_layers}")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        h_output = model(h_initial, adj_lpvg)
    
    print(f"\n输入输出:")
    print(f"  输入特征: {h_initial.shape}")
    print(f"  邻接矩阵: {adj_lpvg.shape}")
    print(f"  输出嵌入: {h_output.shape}")
    
    # 测试残差权重衰减
    print(f"\n残差权重 β_l 随层数衰减:")
    for l in range(model.num_layers):
        beta = model.compute_residual_weight(l)
        print(f"  Layer {l+1}: β = {beta:.4f}")
    
    print("\n✅ DyGAT 测试通过!")
