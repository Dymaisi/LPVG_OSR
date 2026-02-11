"""
训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any
from pathlib import Path
from tqdm import tqdm
import numpy as np

from utils.metrics import compute_osr_metrics


class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        logger,
        output_dirs: Dict[str, Path]
    ):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            config: 配置
            device: 设备
            logger: 日志器
            output_dirs: 输出目录
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.logger = logger
        self.output_dirs = output_dirs
        
        # 训练参数
        self.num_epochs = config.training.num_epochs
        self.learning_rate = config.training.learning_rate
        self.weight_decay = config.training.weight_decay
        
        # 优化器
        if config.training.optimizer == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif config.training.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        
        # 学习率调度器
        if config.training.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.training.step_size,
                gamma=config.training.gamma
            )
        elif config.training.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs
            )
        else:
            self.scheduler = None
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 最佳模型
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train(self):
        """训练模型"""
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 80)
            
            # 训练一个 epoch
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            if epoch % self.config.evaluation.eval_frequency == 0:
                val_loss, val_acc, val_metrics = self.evaluate(
                    self.val_loader,
                    split='val'
                )
                
                # 保存最佳模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"✓ 新的最佳模型! 验证准确率: {val_acc:.4f}")
                
                # 打印指标
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(
                    f"Val Loss: {val_loss:.4f}, "
                    f"Known Acc: {val_metrics['known_acc']:.4f}, "
                    f"Unknown Acc: {val_metrics['unknown_acc']:.4f}, "
                    f"H-Score: {val_metrics['h_score']:.4f}"
                )
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
        
        print(f"\n训练完成! 最佳 epoch: {self.best_epoch}, "
              f"最佳验证准确率: {self.best_val_acc:.4f}")
    
    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        loss_batches = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for features, labels, _ in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            logits, _ = self.model(features)
            loss = self.criterion(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, split: str = 'val'):
        """评估模型"""
        self.model.eval()
        
        total_loss = 0.0
        loss_batches = 0
        all_labels = []
        all_preds = []
        all_scores = []
        all_is_unknown = []
        
        for features, labels, is_unknown in data_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            logits, _ = self.model(features)
            known_mask = ~is_unknown.bool()
            if known_mask.any():
                loss = self.criterion(logits[known_mask], labels[known_mask])
                total_loss += loss.item()
                loss_batches += 1
            
            # 预测
            scores = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            all_is_unknown.extend(is_unknown.numpy())
        
        # 转换为数组
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_scores = np.array(all_scores)
        all_is_unknown = np.array(all_is_unknown, dtype=bool)
        
        # 计算指标
        avg_loss = total_loss / max(loss_batches, 1)
        
        # 已知类准确率
        known_mask = ~all_is_unknown
        known_acc = (all_preds[known_mask] == all_labels[known_mask]).mean()
        
        # 开放集指标
        metrics = compute_osr_metrics(
            labels=all_labels,
            predictions=all_preds,
            scores=all_scores,
            is_unknown=all_is_unknown
        )
        
        metrics['loss'] = avg_loss
        metrics['known_acc'] = known_acc
        
        return avg_loss, known_acc, metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = self.output_dirs['checkpoint_dir'] / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = self.output_dirs['checkpoint_dir'] / 'best.pth'
            torch.save(checkpoint, best_path)
