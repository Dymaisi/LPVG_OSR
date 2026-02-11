"""
生成多转速、多开放度的开放集实验配置文件
"""
import yaml
import os
from pathlib import Path
import random

# 设置随机种子以保证可复现性
random.seed(42)

# 配置输出目录
OUTPUT_DIR = Path(__file__).parent.parent / "config" / "dataset" / "osr_configs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CWRU 配置生成
# ============================================================================

CWRU_RPMS = {
    1797: {"load": "0 HP", "desc": "No load"},
    1772: {"load": "1 HP", "desc": "Light load"},
    1750: {"load": "2 HP", "desc": "Medium load"},
    1730: {"load": "3 HP", "desc": "Heavy load"}
}

# CWRU 故障类型定义（12个类别）
CWRU_FAULT_TYPES = [
    "Normal",
    "Inner_007", "Inner_014", "Inner_021",
    "Outer_007", "Outer_014", "Outer_021",
    "Ball_007", "Ball_014", "Ball_021",
    "Inner_028", "Outer_028"  # 如果数据集包含这些严重程度
]

def generate_cwru_config(rpm, openness_level):
    """
    生成 CWRU 开放集配置
    Args:
        rpm: 转速 (1797, 1772, 1750, 1730)
        openness_level: 开放度等级 ("easy", "medium", "hard")
    """
    # 根据开放度等级设置已知/未知类
    if openness_level == "easy":
        n_known = 9
        n_unknown = 3
        openness = 0.25
    elif openness_level == "medium":
        n_known = 6
        n_unknown = 6
        openness = 0.50
    else:  # hard
        n_known = 4
        n_unknown = 8
        openness = 0.75
    
    # 随机选择已知类和未知类（确保包含 Normal）
    all_classes = list(range(len(CWRU_FAULT_TYPES)))
    
    # Normal (0) 始终作为已知类
    known_classes = [0]
    
    # 从剩余类别中随机选择
    remaining = [i for i in all_classes if i != 0]
    random.shuffle(remaining)
    
    known_classes.extend(remaining[:n_known-1])
    unknown_classes = remaining[n_known-1:n_known-1+n_unknown]
    
    known_classes.sort()
    unknown_classes.sort()
    
    # 构建配置
    config = {
        "dataset": {
            "name": f"cwru_osr_{openness_level}_rpm{rpm}",
            "data_root": "../dataset/CWRU",
            "target_rpm": rpm,
            "motor_load": CWRU_RPMS[rpm]["load"],
            "known_classes": known_classes,
            "unknown_classes": unknown_classes,
            "openness": round(openness, 3),
            "class_labels": {
                i: CWRU_FAULT_TYPES[i] for i in (known_classes + unknown_classes)
            },
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "unknown_ratio": round(openness, 2),
            "sampling_rate": 12000,
            "signal_length": 2048,
            "overlap": 0.5,
            "preprocessing": {
                "normalization": "zscore",
                "filter": None,
                "method": "fft"
            },
            "augmentation": {
                "enable": False
            },
            "notes": f"{openness_level.capitalize()} openness ({int(openness*100)}%) at {rpm} RPM ({CWRU_RPMS[rpm]['load']})"
        }
    }
    
    # 保存配置
    filename = OUTPUT_DIR / f"cwru_{openness_level}_rpm{rpm}.yaml"
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"✓ Created: {filename.name}")
    return config


# ============================================================================
# BDAE 配置生成
# ============================================================================

BDAE_DOMAINS = {
    "E": {
        "rpm_range": "LP=1000-1500 RPM",
        "conditions": [1, 2, 3, 4, 5, 6],
        "representative_rpm": 1200,
        "desc": "Low speed domain"
    },
    "F": {
        "rpm_range": "LP=2000-3000 RPM",
        "conditions": [7, 8, 9, 10, 11, 12, 13, 14],
        "representative_rpm": 2500,
        "desc": "Medium speed domain"
    },
    "G": {
        "rpm_range": "LP=3500-4000 RPM",
        "conditions": [15, 16, 17, 18, 19, 20],
        "representative_rpm": 3750,
        "desc": "High speed domain"
    },
    "H": {
        "rpm_range": "LP=4100-5000 RPM",
        "conditions": [21, 22, 23, 24, 25, 26, 27],
        "representative_rpm": 4500,
        "desc": "Very high speed domain"
    }
}

# BDAE 故障类型（4个基础类别 × 4个域 = 16个类别）
BDAE_BASE_FAULTS = [
    "Normal",
    "Inner_0.5-0.5",
    "Inner_0.5-1.0",
    "Outer_0.5-0.5"
]

def generate_bdae_config(domain, openness_level):
    """
    生成 BDAE 开放集配置
    Args:
        domain: 转速域 ("E", "F", "G", "H")
        openness_level: 开放度等级 ("easy", "medium", "hard")
    """
    # 根据开放度等级设置已知/未知类
    # 对于 BDAE，我们在单个域内划分已知/未知故障类型
    if openness_level == "easy":
        # 已知3个故障类型，未知1个
        n_known_faults = 3
        n_unknown_faults = 1
        openness = 0.25
    elif openness_level == "medium":
        # 已知2个，未知2个
        n_known_faults = 2
        n_unknown_faults = 2
        openness = 0.50
    else:  # hard
        # 已知1个（Normal），未知3个
        n_known_faults = 1
        n_unknown_faults = 3
        openness = 0.75
    
    # 随机选择已知/未知故障类型
    all_fault_indices = list(range(len(BDAE_BASE_FAULTS)))
    
    # Normal (0) 始终作为已知类
    if openness_level == "hard":
        known_fault_indices = [0]
    else:
        known_fault_indices = [0]
        remaining = [i for i in all_fault_indices if i != 0]
        random.shuffle(remaining)
        known_fault_indices.extend(remaining[:n_known_faults-1])
    
    unknown_fault_indices = [i for i in all_fault_indices if i not in known_fault_indices][:n_unknown_faults]
    
    # 构建类别标签
    known_classes = known_fault_indices
    unknown_classes = [len(BDAE_BASE_FAULTS) + i for i in range(len(unknown_fault_indices))]
    
    class_labels = {}
    for idx, fault_idx in enumerate(known_fault_indices):
        class_labels[idx] = f"{BDAE_BASE_FAULTS[fault_idx]}_Domain{domain}"
    
    for idx, fault_idx in enumerate(unknown_fault_indices):
        class_labels[len(known_fault_indices) + idx] = f"{BDAE_BASE_FAULTS[fault_idx]}_Domain{domain}"
    
    # 构建配置
    config = {
        "dataset": {
            "name": f"bdae_osr_{openness_level}_domain{domain}",
            "data_root": "../dataset/drive-download-20251103T072647Z-1-001",
            "target_domain": domain,
            "domain_info": BDAE_DOMAINS[domain],
            "known_classes": known_classes,
            "unknown_classes": unknown_classes,
            "openness": round(openness, 3),
            "class_labels": class_labels,
            "working_conditions": BDAE_DOMAINS[domain]["conditions"],
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "unknown_ratio": round(openness, 2),
            "sampling_rate": 25600,
            "signal_length": 2048,
            "overlap": 0.5,
            "use_channels": [0, 1, 2, 3, 4, 5],
            "preprocessing": {
                "normalization": "zscore",
                "filter": None,
                "method": "fft"
            },
            "augmentation": {
                "enable": False
            },
            "notes": f"{openness_level.capitalize()} openness ({int(openness*100)}%) at Domain {domain} ({BDAE_DOMAINS[domain]['rpm_range']})"
        }
    }
    
    # 保存配置
    filename = OUTPUT_DIR / f"bdae_{openness_level}_domain{domain}.yaml"
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"✓ Created: {filename.name}")
    return config


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("生成多转速、多开放度的开放集实验配置")
    print("=" * 70)
    
    # 生成 CWRU 配置 (4 转速 × 3 开放度 = 12 个)
    print("\n📁 生成 CWRU 配置...")
    print("-" * 70)
    cwru_configs = []
    for rpm in CWRU_RPMS.keys():
        for openness in ["easy", "medium", "hard"]:
            config = generate_cwru_config(rpm, openness)
            cwru_configs.append(config)
    
    # 生成 BDAE 配置 (4 域 × 3 开放度 = 12 个)
    print("\n📁 生成 BDAE 配置...")
    print("-" * 70)
    bdae_configs = []
    for domain in BDAE_DOMAINS.keys():
        for openness in ["easy", "medium", "hard"]:
            config = generate_bdae_config(domain, openness)
            bdae_configs.append(config)
    
    print("\n" + "=" * 70)
    print(f"✅ 完成！共生成 {len(cwru_configs) + len(bdae_configs)} 个配置文件")
    print(f"   - CWRU: {len(cwru_configs)} 个 (4 转速 × 3 开放度)")
    print(f"   - BDAE: {len(bdae_configs)} 个 (4 域 × 3 开放度)")
    print(f"   - 保存位置: {OUTPUT_DIR}")
    print("=" * 70)
    
    # 生成实验脚本
    generate_experiment_script(cwru_configs, bdae_configs)


def generate_experiment_script(cwru_configs, bdae_configs):
    """生成批量实验运行脚本"""
    script_content = """#!/bin/bash
# 批量运行所有开放集实验配置

echo "========================================="
echo "多转速、多开放度开放集实验"
echo "========================================="

# CWRU 实验 (4 转速 × 3 开放度 = 12 个)
echo ""
echo "📊 CWRU 数据集实验..."
echo "-----------------------------------------"

for rpm in 1797 1772 1750 1730; do
    for openness in easy medium hard; do
        config="LPVG_OSR/config/dataset/osr_configs/cwru_${openness}_rpm${rpm}.yaml"
        echo "▶ Running: $config"
        python run_experiments.py --config $config
    done
done

# BDAE 实验 (4 域 × 3 开放度 = 12 个)
echo ""
echo "📊 BDAE 数据集实验..."
echo "-----------------------------------------"

for domain in E F G H; do
    for openness in easy medium hard; do
        config="LPVG_OSR/config/dataset/osr_configs/bdae_${openness}_domain${domain}.yaml"
        echo "▶ Running: $config"
        python run_experiments.py --config $config
    done
done

echo ""
echo "========================================="
echo "✅ 所有实验完成！"
echo "========================================="
"""
    
    script_path = OUTPUT_DIR.parent.parent / "scripts" / "run_all_osr_experiments.sh"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"\n📝 实验运行脚本已生成: {script_path}")


if __name__ == "__main__":
    main()
