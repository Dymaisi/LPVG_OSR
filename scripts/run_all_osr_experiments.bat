@echo off
REM ============================================================================
REM 批量运行所有开放集实验配置
REM Windows PowerShell 版本
REM ============================================================================

echo =========================================
echo 多转速、多开放度开放集实验
echo =========================================

REM ============================================================================
REM CWRU 实验 (4 转速 x 3 开放度 = 12 个)
REM ============================================================================
echo.
echo [CWRU 数据集实验]
echo -----------------------------------------

for %%r in (1797 1772 1750 1730) do (
    for %%o in (easy medium hard) do (
        echo.
        echo ^>^> 运行: CWRU %%o @ %%r RPM
        python run_experiments.py --config LPVG_OSR/config/dataset/osr_configs/cwru_%%o_rpm%%r.yaml --output_dir results/cwru_%%r_%%o
    )
)

REM ============================================================================
REM BDAE 实验 (4 域 x 3 开放度 = 12 个)
REM ============================================================================
echo.
echo [BDAE 数据集实验]
echo -----------------------------------------

for %%d in (E F G H) do (
    for %%o in (easy medium hard) do (
        echo.
        echo ^>^> 运行: BDAE %%o @ Domain %%d
        python run_experiments.py --config LPVG_OSR/config/dataset/osr_configs/bdae_%%o_domain%%d.yaml --output_dir results/bdae_%%d_%%o
    )
)

echo.
echo =========================================
echo 所有实验完成！
echo =========================================
pause
