"""
日志工具
"""

import logging
from pathlib import Path
from datetime import datetime


def setup_logger(log_dir: Path, experiment_name: str) -> logging.Logger:
    """设置日志器
    
    Args:
        log_dir: 日志目录
        experiment_name: 实验名称
        
    Returns:
        logger: 日志器
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志器
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    logger.handlers = []
    
    # 文件处理器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志保存到: {log_file}")
    
    return logger
