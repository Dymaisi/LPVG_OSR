#!/bin/bash
# ============================================================================
# 批量运行所有开放集实验配置
# Linux/Mac 版本
# ============================================================================

echo "========================================="
echo "多转速、多开放度开放集实验"
echo "========================================="

# ============================================================================
# CWRU 实验 (4 转速 × 3 开放度 = 12 个)
# ============================================================================
echo ""
echo "📊 CWRU 数据集实验..."
echo "-----------------------------------------"

for rpm in 1797 1772 1750 1730; do
    for openness in easy medium hard; do
        config="LPVG_OSR/config/dataset/osr_configs/cwru_${openness}_rpm${rpm}.yaml"
        echo ""
        echo "▶ Running: CWRU ${openness} @ ${rpm} RPM"
        python run_experiments.py \
            --config $config \
            --output_dir results/cwru_${rpm}_${openness}
    done
done

# ============================================================================
# BDAE 实验 (4 域 × 3 开放度 = 12 个)
# ============================================================================
echo ""
echo "📊 BDAE 数据集实验..."
echo "-----------------------------------------"

for domain in E F G H; do
    for openness in easy medium hard; do
        config="LPVG_OSR/config/dataset/osr_configs/bdae_${openness}_domain${domain}.yaml"
        echo ""
        echo "▶ Running: BDAE ${openness} @ Domain ${domain}"
        python run_experiments.py \
            --config $config \
            --output_dir results/bdae_${domain}_${openness}
    done
done

echo ""
echo "========================================="
echo "✅ 所有实验完成！"
echo "========================================="
