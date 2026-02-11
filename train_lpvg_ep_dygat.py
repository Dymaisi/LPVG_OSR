"""
LPVG-EP-DyGAT 完整训练脚本
严格按照论文方法论实现
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import numpy as np
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from tqdm import tqdm
import argparse

# 导入自定义模块
from models.lpvg_ep_dygat import create_lpvg_ep_dygat
from models.evidential_classifier import EvidentialLoss, open_set_prediction
from data.lpvg_builder_enhanced import EnhancedLPVGBuilder, extract_graph_features
# 直接导入需要的指标计算函数，避免 omegaconf 依赖
import sklearn.metrics as skmetrics


def compute_osr_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainties: np.ndarray,
    known_classes: List[int] = None
) -> Dict[str, float]:
    """计算开放集识别指标"""
    if known_classes is None:
        known_classes = list(set(y_true) - {max(y_true)})
    
    # 已知类掩码
    known_mask = np.isin(y_true, known_classes)
    unknown_mask = ~known_mask
    
    # 已知类准确率
    if known_mask.sum() > 0:
        known_acc = skmetrics.accuracy_score(
            y_true[known_mask],
            y_pred[known_mask]
        )
    else:
        known_acc = 0.0
    
    # 未知类准确率（预测为 -1 的比例）
    if unknown_mask.sum() > 0:
        unknown_acc = (y_pred[unknown_mask] == -1).sum() / unknown_mask.sum()
    else:
        unknown_acc = 1.0
    
    # H-score
    if known_acc + unknown_acc > 0:
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc)
    else:
        h_score = 0.0
    
    # AUROC（已知 vs 未知）
    try:
        binary_labels = (~known_mask).astype(int)
        auroc = skmetrics.roc_auc_score(binary_labels, uncertainties)
    except:
        auroc = 0.0
    
    # 总体准确率
    accuracy = skmetrics.accuracy_score(
        y_true[known_mask],
        y_pred[known_mask]
    ) if known_mask.sum() > 0 else 0.0
    
    return {
        'known_acc': known_acc,
        'unknown_acc': unknown_acc,
        'h_score': h_score,
        'auroc': auroc,
        'accuracy': accuracy
    }


class LPVGOSRDataset(Dataset):
    """LPVG 开放集识别数据集"""
    
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        lpvg_builder: EnhancedLPVGBuilder,
        graphs: List[nx.Graph] = None,
        build_graphs: bool = True
    ):
        """
        Args:
            signals: (N, L) 信号数组
            labels: (N,) 标签数组
            lpvg_builder: LPVG 构建器
            graphs: 预构建的图列表
            build_graphs: 是否构建图
        """
        self.signals = signals
        self.labels = labels
        self.lpvg_builder = lpvg_builder
        
        if graphs is not None:
            self.graphs = graphs
        elif build_graphs:
            print(f"构建 {len(signals)} 个 LPVG 图...")
            self.graphs = lpvg_builder.build_batch(signals, show_progress=True)
        else:
            self.graphs = None
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = self.labels[idx]
        
        # 提取节点特征和邻接矩阵
        node_features = extract_graph_features(graph)
        adj_matrix = nx.to_numpy_array(graph)
        
        # 添加自环（论文中的标准做法）
        adj_matrix = adj_matrix + np.eye(len(adj_matrix))
        
        return {
            'node_features': torch.FloatTensor(node_features),
            'adj_matrix': torch.FloatTensor(adj_matrix),
            'label': torch.LongTensor([label])[0]
        }


def collate_fn(batch):
    """自定义批处理函数（处理不同大小的图）"""
    # 找到最大节点数
    max_nodes = max([item['node_features'].shape[0] for item in batch])
    feature_dim = batch[0]['node_features'].shape[1]
    batch_size = len(batch)
    
    # 初始化批次张量
    batch_node_features = torch.zeros(batch_size, max_nodes, feature_dim)
    batch_adj_matrix = torch.zeros(batch_size, max_nodes, max_nodes)
    batch_labels = torch.zeros(batch_size, dtype=torch.long)
    
    # 填充数据
    for i, item in enumerate(batch):
        num_nodes = item['node_features'].shape[0]
        batch_node_features[i, :num_nodes, :] = item['node_features']
        batch_adj_matrix[i, :num_nodes, :num_nodes] = item['adj_matrix']
        batch_labels[i] = item['label']
    
    return {
        'node_features': batch_node_features,
        'adj_matrix': batch_adj_matrix,
        'label': batch_labels
    }


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: EvidentialLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """训练一个 epoch"""
    model.train()
    
    total_loss = 0.0
    loss_components = {'loss_edl': 0.0, 'loss_kl': 0.0, 'loss_proto': 0.0}
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        # 数据移到设备
        node_features = batch['node_features'].to(device)
        adj_matrix = batch['adj_matrix'].to(device)
        labels = batch['label'].to(device)
        
        batch_size = labels.size(0)
        
        # 前向传播（逐样本处理）
        all_outputs = []
        all_graph_embeddings = []
        
        for i in range(batch_size):
            outputs = model(node_features[i], adj_matrix[i])
            all_outputs.append(outputs)
            all_graph_embeddings.append(outputs['graph_embedding'])
        
        # 合并输出
        graph_embeddings = torch.stack(all_graph_embeddings, dim=0)
        alphas = torch.cat([o['alpha'] for o in all_outputs], dim=0)
        S = torch.cat([o['S'] for o in all_outputs], dim=0)
        beliefs = torch.cat([o['belief'] for o in all_outputs], dim=0)
        
        merged_outputs = {
            'alpha': alphas,
            'S': S,
            'belief': beliefs
        }
        
        # 计算损失
        loss, loss_dict = criterion(
            merged_outputs,
            graph_embeddings,
            labels,
            model.get_prototypes(),
            epoch=epoch
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict.get(key, 0.0)
        
        # 计算准确率
        preds = torch.argmax(beliefs, dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.0 * correct / total:.2f}%"
        })
    
    # 计算平均值
    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'accuracy': 100.0 * correct / total
    }
    
    for key in loss_components:
        metrics[key] = loss_components[key] / num_batches
    
    return metrics


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    uncertainty_threshold: float = 0.5,
    known_classes: List[int] = None
) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_uncertainties = []
    all_beliefs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            node_features = batch['node_features'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            labels = batch['label'].to(device)
            
            batch_size = labels.size(0)
            
            # 前向传播
            for i in range(batch_size):
                outputs = model(node_features[i], adj_matrix[i])
                
                predictions, is_unknown = open_set_prediction(
                    outputs, 
                    uncertainty_threshold
                )
                
                all_preds.append(predictions[0].item())
                all_labels.append(labels[i].item())
                all_uncertainties.append(outputs['uncertainty'][0].item())
                all_beliefs.append(outputs['belief'][0].cpu().numpy())
    
    # 转换为 numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_uncertainties = np.array(all_uncertainties)
    
    # 计算 OSR 指标
    metrics = compute_osr_metrics(
        all_labels,
        all_preds,
        all_uncertainties,
        known_classes=known_classes
    )
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='LPVG-EP-DyGAT 训练脚本')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_samples', type=int, default=300, help='样本数量')
    parser.add_argument('--signal_length', type=int, default=512, help='信号长度')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    args = parser.parse_args()
    
    print("=" * 80)
    print("LPVG-EP-DyGAT 完整训练流程")
    print("=" * 80)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 生成模拟数据
    print(f"\n[1/6] 生成模拟数据")
    np.random.seed(42)
    
    # 训练集：3 个已知类
    train_signals = []
    train_labels = []
    
    for cls in range(3):
        for _ in range(args.num_samples // 3):
            t = np.linspace(0, 1, args.signal_length)
            # 不同类别有不同的频率特征
            freq1 = 10 + cls * 5
            freq2 = 30 + cls * 10
            signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
            signal += 0.3 * np.random.randn(args.signal_length)
            
            train_signals.append(signal)
            train_labels.append(cls)
    
    train_signals = np.array(train_signals)
    train_labels = np.array(train_labels)
    
    # 测试集：3 个已知类 + 1 个未知类
    test_signals = []
    test_labels = []
    
    # 已知类
    for cls in range(3):
        for _ in range(30):
            t = np.linspace(0, 1, args.signal_length)
            freq1 = 10 + cls * 5
            freq2 = 30 + cls * 10
            signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
            signal += 0.3 * np.random.randn(args.signal_length)
            
            test_signals.append(signal)
            test_labels.append(cls)
    
    # 未知类（标记为 3）
    for _ in range(30):
        t = np.linspace(0, 1, args.signal_length)
        # 完全不同的频率特征
        signal = np.sin(2 * np.pi * 50 * t) + np.cos(2 * np.pi * 80 * t)
        signal += 0.5 * np.random.randn(args.signal_length)
        
        test_signals.append(signal)
        test_labels.append(3)  # 未知类
    
    test_signals = np.array(test_signals)
    test_labels = np.array(test_labels)
    
    print(f"  训练集: {train_signals.shape}, 已知类: {np.unique(train_labels)}")
    print(f"  测试集: {test_signals.shape}, 已知类: {np.unique(test_labels[:90])}, 未知类: 3")
    
    # 构建 LPVG 图
    print(f"\n[2/6] 构建 LPVG 图（P=2）")
    lpvg_builder = EnhancedLPVGBuilder(
        P=2,  # 论文推荐参数
        local_window=5,
        use_zscore=True,
        use_instantaneous_freq=True
    )
    
    train_dataset = LPVGOSRDataset(train_signals, train_labels, lpvg_builder)
    test_dataset = LPVGOSRDataset(test_signals, test_labels, lpvg_builder)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  测试批次数: {len(test_loader)}")
    
    # 创建模型
    print(f"\n[3/6] 创建 LPVG-EP-DyGAT 模型")
    config = {
        'node_feature_dim': 4,
        'dygat_hidden_dims': [64, 64, 32],
        'dygat_num_heads': 8,
        'dygat_dropout': 0.3,
        'lambda_residual': 2.0,
        'use_dynamic_graph': True,
        'top_k': 10,
        'mu': 0.5,
        'num_classes': 3,
        'sigma': 1.0,
        'learn_prototypes': True,
        'global_pooling': 'mean'
    }
    
    model = create_lpvg_ep_dygat(config).to(device)
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建损失函数和优化器
    criterion = EvidentialLoss(
        num_classes=3,
        lambda_kl=0.1,
        lambda_proto=0.01,
        kl_annealing=True
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 训练循环
    print(f"\n[4/6] 开始训练（{args.epochs} epochs）")
    best_h_score = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 评估（每 5 个 epoch）
        if epoch % 5 == 0:
            test_metrics = evaluate(
                model, test_loader, device,
                uncertainty_threshold=0.5,
                known_classes=[0, 1, 2]
            )
            
            print(f"\nEpoch {epoch}/{args.epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Test  - Known Acc: {test_metrics['known_acc']:.4f}, "
                  f"Unknown Acc: {test_metrics['unknown_acc']:.4f}, "
                  f"H-score: {test_metrics['h_score']:.4f}")
            
            # 保存最佳模型
            if test_metrics['h_score'] > best_h_score:
                best_h_score = test_metrics['h_score']
                torch.save(model.state_dict(), 'best_lpvg_ep_dygat.pth')
                print(f"  ✓ 新的最佳 H-score: {best_h_score:.4f}")
        
        scheduler.step()
    
    # 最终评估
    print(f"\n[5/6] 最终评估")
    model.load_state_dict(torch.load('best_lpvg_ep_dygat.pth'))
    final_metrics = evaluate(
        model, test_loader, device,
        uncertainty_threshold=0.5,
        known_classes=[0, 1, 2]
    )
    
    print("\n" + "=" * 80)
    print("最终评估结果")
    print("=" * 80)
    print(f"\n【核心 OSR 指标】")
    print(f"  已知类准确率 (Known Acc):   {final_metrics['known_acc']:.4f}")
    print(f"  未知类准确率 (Unknown Acc): {final_metrics['unknown_acc']:.4f}")
    print(f"  H-score:                    {final_metrics['h_score']:.4f}")
    print(f"\n【其他指标】")
    print(f"  AUROC:                      {final_metrics.get('auroc', 0.0):.4f}")
    print(f"  Overall Accuracy:           {final_metrics.get('accuracy', 0.0):.4f}")
    
    print("\n✅ LPVG-EP-DyGAT 训练完成!")


if __name__ == "__main__":
    main()
