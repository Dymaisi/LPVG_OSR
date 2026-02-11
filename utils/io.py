"""
文件 IO 工具
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf


def save_config(config: DictConfig, save_path: Path) -> None:
    """保存配置到 YAML 文件
    
    Args:
        config: 配置对象
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(OmegaConf.to_container(config), f, allow_unicode=True)
    
    print(f"配置已保存到: {save_path}")


def create_output_dirs(config: DictConfig) -> Dict[str, Path]:
    """创建输出目录
    
    Args:
        config: 配置对象
        
    Returns:
        output_dirs: 输出目录字典
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.experiment_name}_{timestamp}"
    
    output_dirs = {
        'checkpoint_dir': Path(config.output.checkpoint_dir) / exp_name,
        'log_dir': Path(config.logging.log_dir) / exp_name,
        'visualization_dir': Path(config.output.visualization_dir) / exp_name,
        'result_dir': Path(config.output.result_dir) / exp_name
    }
    
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("\n输出目录:")
    for name, path in output_dirs.items():
        print(f"  {name}: {path}")
    
    return output_dirs
