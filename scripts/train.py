"""
训练脚本
"""

import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from data import CWRUOpenSet, BDAEOpenSet, LPVGBuilder
from data.lpvg_builder_enhanced import EnhancedLPVGBuilder
from data.graph_utils import GraphBatchCollator
from models import LPVG_MLP, create_lpvg_ep_dygat
from core.trainer import Trainer
from core.trainer_evidential import EvidentialTrainer
from utils.logger import setup_logger
from utils.io import save_config, create_output_dirs


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    """主训练函数
    
    Args:
        cfg: Hydra 配置
    """
    # 打印配置
    print("="*80)
    print("实验配置:")
    print("="*80)
    print(OmegaConf.to_yaml(cfg))
    print("="*80)
    
    # 设置随机种子
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # 设置设备
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建输出目录
    output_dirs = create_output_dirs(cfg)
    
    # 设置日志
    logger = setup_logger(
        log_dir=output_dirs['log_dir'],
        experiment_name=cfg.experiment_name
    )
    logger.info(f"开始实验: {cfg.experiment_name}")
    
    # 保存配置
    save_config(cfg, output_dirs['checkpoint_dir'] / 'config.yaml')
    
    # 1. 归一化数据集配置（兼容 osr_configs 的 dataset: 包装）
    dataset_cfg = cfg.dataset
    if isinstance(dataset_cfg, DictConfig) and 'dataset' in dataset_cfg:
        dataset_cfg = dataset_cfg['dataset']

    model_cfg = cfg.model
    if isinstance(model_cfg, DictConfig) and 'model' in model_cfg:
        model_cfg = model_cfg['model']

    method_cfg = cfg.method
    if isinstance(method_cfg, DictConfig) and 'method' in method_cfg:
        method_cfg = method_cfg['method']

    # 2. 加载数据集
    print("\n" + "="*80)
    print("加载数据集")
    print("="*80)
    
    dataset_name = dataset_cfg.name.lower()

    if 'cwru' in dataset_name:
        train_dataset = CWRUOpenSet(OmegaConf.to_container(dataset_cfg), split='train')
        val_dataset = CWRUOpenSet(OmegaConf.to_container(dataset_cfg), split='val')
        test_dataset = CWRUOpenSet(OmegaConf.to_container(dataset_cfg), split='test')
    elif 'bdae' in dataset_name:
        train_dataset = BDAEOpenSet(OmegaConf.to_container(dataset_cfg), split='train')
        val_dataset = BDAEOpenSet(OmegaConf.to_container(dataset_cfg), split='val')
        test_dataset = BDAEOpenSet(OmegaConf.to_container(dataset_cfg), split='test')
    else:
        raise ValueError(f"未知数据集: {dataset_cfg.name}")
    
    print(f"\n训练集信息: {train_dataset.get_info()}")
    print(f"验证集信息: {val_dataset.get_info()}")
    print(f"测试集信息: {test_dataset.get_info()}")
    
    # 2. 构建 LPVG 图
    print("\n" + "="*80)
    print("构建 LPVG 图")
    print("="*80)
    
    lpvg_m = cfg.lpvg.get('m', 1)
    use_cache = cfg.lpvg.get('cache_graphs', True)
    
    # 3. 创建数据加载器（模型相关）
    from torch.utils.data import DataLoader, TensorDataset, Dataset

    if model_cfg.name == 'lpvg_ep_dygat':
        lpvg_params = model_cfg.get('lpvg', {})
        max_nodes = lpvg_params.get('max_nodes', None)
        builder = EnhancedLPVGBuilder(
            P=lpvg_params.get('P', 2),
            local_window=lpvg_params.get('local_window', 5),
            use_zscore=lpvg_params.get('use_zscore', True),
            use_instantaneous_freq=lpvg_params.get('use_instantaneous_freq', True),
            max_nodes=max_nodes,
            use_cache=use_cache,
            cache_dir=cfg.lpvg.get('cache_dir', '.cache/lpvg_enhanced')
        )

        graph_batch_size = model_cfg.get('batch_size', cfg.training.batch_size)

        class GraphDataset(Dataset):
            def __init__(self, graphs, labels, is_unknown):
                self.graphs = graphs
                self.labels = labels
                self.is_unknown = is_unknown

            def __len__(self):
                return len(self.graphs)

            def __getitem__(self, idx):
                return self.graphs[idx], int(self.labels[idx]), int(self.is_unknown[idx])

        def build_graphs(dataset, cache_name: str):
            if use_cache:
                cached = builder.load_cache(cache_name)
                if cached is not None:
                    return cached
            graphs = builder.build_batch(dataset.signals, show_progress=True)
            if use_cache:
                builder.save_cache(graphs, cache_name)
            return graphs

        cache_tag = f"{dataset_cfg.name}_P{lpvg_params.get('P', 2)}_n{max_nodes if max_nodes else 'full'}"

        print("训练集...")
        train_graphs = build_graphs(train_dataset, f"{cache_tag}_train")
        print("验证集...")
        val_graphs = build_graphs(val_dataset, f"{cache_tag}_val")
        print("测试集...")
        test_graphs = build_graphs(test_dataset, f"{cache_tag}_test")

        train_loader = DataLoader(
            GraphDataset(train_graphs, train_dataset.labels, train_dataset.is_unknown),
            batch_size=graph_batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.num_workers,
            collate_fn=GraphBatchCollator(add_self_loops=True)
        )
        val_loader = DataLoader(
            GraphDataset(val_graphs, val_dataset.labels, val_dataset.is_unknown),
            batch_size=graph_batch_size,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
            collate_fn=GraphBatchCollator(add_self_loops=True)
        )
        test_loader = DataLoader(
            GraphDataset(test_graphs, test_dataset.labels, test_dataset.is_unknown),
            batch_size=graph_batch_size,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
            collate_fn=GraphBatchCollator(add_self_loops=True)
        )
    else:
        builder = LPVGBuilder(m=lpvg_m, use_cache=use_cache)

        print("训练集...")
        train_dataset.build_lpvg_graphs(m=lpvg_m, use_cache=use_cache)
        train_features = train_dataset.extract_graph_features()

        print("验证集...")
        val_dataset.build_lpvg_graphs(m=lpvg_m, use_cache=use_cache)
        val_features = val_dataset.extract_graph_features()

        print("测试集...")
        test_dataset.build_lpvg_graphs(m=lpvg_m, use_cache=use_cache)
        test_features = test_dataset.extract_graph_features()

        print(f"\n图特征维度: {train_features.shape[1]}")

        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(train_features),
                torch.LongTensor(train_dataset.labels),
                torch.FloatTensor(train_dataset.is_unknown)
            ),
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.num_workers
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(val_features),
                torch.LongTensor(val_dataset.labels),
                torch.FloatTensor(val_dataset.is_unknown)
            ),
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(test_features),
                torch.LongTensor(test_dataset.labels),
                torch.FloatTensor(test_dataset.is_unknown)
            ),
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers
        )
    
    # 4. 创建模型
    print("\n" + "="*80)
    print("创建模型")
    print("="*80)
    
    model_config = OmegaConf.to_container(model_cfg)
    model_config['num_classes'] = len(dataset_cfg.known_classes)

    if model_cfg.name == 'lpvg_ep_dygat':
        model = create_lpvg_ep_dygat(model_config).to(device)
    elif model_cfg.name == 'lpvg_mlp':
        model_config['graph_features'] = {'feature_dim': train_features.shape[1]}
        model = LPVG_MLP(model_config).to(device)
    else:
        raise ValueError(f"未知模型: {model_cfg.name}")
    
    print(f"模型: {model.__class__.__name__}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. 训练
    print("\n" + "="*80)
    print("开始训练")
    print("="*80)
    
    if model_cfg.name == 'lpvg_ep_dygat':
        trainer = EvidentialTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=cfg,
            method_cfg=OmegaConf.to_container(method_cfg),
            device=device,
            logger=logger,
            output_dirs=output_dirs
        )
    else:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=cfg,
            device=device,
            logger=logger,
            output_dirs=output_dirs
        )
    
    trainer.train()
    
    # 6. 最终评估
    print("\n" + "="*80)
    print("最终评估")
    print("="*80)
    
    trainer.evaluate(test_loader, split='test')
    
    logger.info("实验完成!")


if __name__ == '__main__':
    main()
