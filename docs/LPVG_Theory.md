# LPVG 理论说明

## Limited Penetrable Visibility Graph (LPVG)

### 1. 基本概念

LPVG 是 Visibility Graph (VG) 的扩展版本，允许视线穿透有限数量的中间点。

### 2. 构建规则

对于时间序列 $\{x_i\}_{i=1}^N$，节点 $i$ 和节点 $j$ 之间存在边当且仅当：

$$
x_k < x_i + (x_j - x_i) \frac{k - i}{j - i}, \quad \forall k \in (i, j)
$$

允许最多 $m$ 个中间点违反这个条件。

### 3. 参数说明

- **m = 0**: 标准 Visibility Graph，不允许穿透
- **m = 1**: 允许穿透 1 个点，图更密集
- **m = 2**: 允许穿透 2 个点，图更密集

### 4. 物理意义

- **节点**: 时间序列的采样点
- **边**: 两个时间点之间存在"可见性"
- **穿透**: 允许视线被部分遮挡

### 5. 图特征

从 LPVG 图中可以提取以下拓扑特征：

1. **基本特征**
   - 节点数 (num_nodes)
   - 边数 (num_edges)

2. **度特征**
   - 平均度 (avg_degree)
   - 最大度 (max_degree)
   - 度标准差 (std_degree)

3. **密度特征**
   - 图密度 (density)

4. **聚类特征**
   - 平均聚类系数 (avg_clustering)
   - 传递性 (transitivity)

5. **相关性特征**
   - 度相关性 (assortativity)

6. **距离特征**
   - 直径 (diameter)
   - 平均最短路径 (avg_shortest_path)

### 6. 故障诊断应用

不同故障模式的振动信号会产生不同拓扑结构的 LPVG 图：

- **健康状态**: 规则的周期性信号 → 规则的图结构
- **内圈故障**: 冲击性信号 → 高聚类系数
- **外圈故障**: 局部异常 → 高度不均匀
- **滚珠故障**: 随机性强 → 低密度

### 7. 参考文献

1. Lacasa, L., et al. (2008). "From time series to complex networks: The visibility graph." PNAS.
2. Zhou, T. T., et al. (2012). "Limited penetrable visibility graph for establishing complex network from time series." Acta Physica Sinica.
