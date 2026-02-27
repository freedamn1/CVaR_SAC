import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_train_updates(file_path):
    """
    解析 train_updates.txt 文件
    期望每行格式类似:
    [train] t=  1000 | LossPi= 0.1234 ... | WinRewMean= 12.34 | WinCostMean= 0.5678 ...
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith("[train]"):
                continue
            
            # 去掉 "[train]" 前缀，按 "|" 分割
            parts = line.replace("[train]", "").split("|")
            row = {}
            for part in parts:
                kv = part.strip().split("=")
                if len(kv) == 2:
                    key = kv[0].strip()
                    try:
                        val = float(kv[1].strip())
                        row[key] = val
                    except ValueError:
                        pass
            if row:
                data.append(row)
    
    return pd.DataFrame(data)

def load_data(base_dir, alg_name):
    """
    读取指定算法目录下所有随机种子的 train_updates.txt
    假设目录结构: base_dir/alg_name_s*/train_updates.txt
    """
    # 匹配模式：base_dir 下面所有包含 alg_name 的子目录
    pattern = os.path.join(base_dir, f"*{alg_name}*")
    exp_dirs = glob.glob(pattern)
    
    all_df = []
    for exp_dir in exp_dirs:
        # 找到该实验目录下的所有 seed 文件夹 (通常是 _s0, _s1, ...)
        # 这里假设 exp_dir 本身就是一个 seed 的运行目录，或者包含多个 seed 子目录
        # 根据 WCSAC 的结构，通常是 .../exp_name_s0/train_updates.txt
        
        # 尝试直接找 train_updates.txt
        log_file = os.path.join(exp_dir, "train_updates.txt")
        if os.path.exists(log_file):
            df = parse_train_updates(log_file)
            if not df.empty:
                df['seed'] = exp_dir  # 用路径区分 seed
                all_df.append(df)
        
    if not all_df:
        print(f"Warning: No data found for {alg_name} in {base_dir}")
        return pd.DataFrame()
        
    return pd.concat(all_df, ignore_index=True)

# 全局颜色定义，确保统一
COLORS = {
    'WCSAC (α=0.1)': 'lightcoral', 
    'WCSAC (α=0.5)': 'red', 
    'WCSAC (α=0.9)': 'darkred',
    'SAC-Lagrangian': 'blue', 
    'Fixed-Price': 'green', 
    'Standard-SAC': 'orange'
}

def plot_performance_curve(data_dict, x_col='t', y_col='WinRewMean', ylabel='Average Reward', title='Performance', ax=None, save_path=None):
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        is_standalone = True
    else:
        is_standalone = False
        
    sns.set_style("whitegrid")
    
    for alg_name, df in data_dict.items():
        if df.empty:
            continue
        # 使用 seaborn 的 lineplot 自动处理均值和置信区间（阴影）
        # 注意：这里直接用传入的ax绘制
        sns.lineplot(data=df, x=x_col, y=y_col, label=alg_name, color=COLORS.get(alg_name, None), linewidth=2, ax=ax)

    ax.set_xlabel('TotalTimeSteps (1e6)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    # 统一图例位置，避免遮挡
    ax.legend(fontsize=10, loc='best')
    
    if is_standalone:
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

def plot_cvar_curve(data_dict, ax=None, save_path=None):
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        is_standalone = True
    else:
        is_standalone = False

    sns.set_style("whitegrid")
    has_cvar_data = False
    
    # 绘制三种 WCSAC 的 CVaR
    for name in ['WCSAC (α=0.1)', 'WCSAC (α=0.5)', 'WCSAC (α=0.9)']:
        if name in data_dict:
            df = data_dict[name]
            if not df.empty and 'QcPiCVaR' in df.columns:
                sns.lineplot(data=df, x='t', y='QcPiCVaR', label=f'{name} CVaR', color=COLORS.get(name, 'red'), linewidth=2, ax=ax)
                has_cvar_data = True

    # 绘制 SAC-Lagrangian 的 Expected Cost (作为对比)
    if 'SAC-Lagrangian' in data_dict:
        saclag_df = data_dict['SAC-Lagrangian']
        if not saclag_df.empty and 'QcPi' in saclag_df.columns:
            sns.lineplot(data=saclag_df, x='t', y='QcPi', label='SAC-Lagrangian ExpCost', color='blue', linestyle='--', linewidth=2, ax=ax)
            has_cvar_data = True
    
    if has_cvar_data:
        ax.set_xlabel('TotalTimeSteps (1e6)', fontsize=12)
        ax.set_ylabel('Risk / Cost Estimate', fontsize=12)
        ax.set_title('Risk (CVaR) vs Expected Cost', fontsize=14)
        ax.legend(fontsize=10)
        
        if is_standalone:
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved CVaR/Cost comparison plot to {save_path}")
    else:
        print("No CVaR/Cost data found to plot.")


def load_data_from_config(config, downsample_factor=1):
    """
    根据配置字典加载数据
    config: { 'AlgName': ['path1', 'path2'] or 'path' }
    downsample_factor: 降采样因子，>=1 的整数。例如设为 2，则每 2 个点取平均，间隔变为原来的 2 倍。
    """
    data_dict = {}
    
    for alg_name, paths in config.items():
        if isinstance(paths, str):
            paths = [paths]
            
        df_list = []
        for p in paths:
            # 如果路径直接是文件
            if os.path.isfile(p) and p.endswith("train_updates.txt"):
                df = parse_train_updates(p)
                if not df.empty:
                    df['seed'] = os.path.dirname(p)
                    df_list.append(df)
            # 如果路径是目录
            elif os.path.isdir(p):
                # 尝试直接加载目录下的 train_updates.txt
                direct_file = os.path.join(p, "train_updates.txt")
                if os.path.exists(direct_file):
                    df = parse_train_updates(direct_file)
                    if not df.empty:
                        df['seed'] = p
                        df_list.append(df)
                else:
                    # 递归查找该目录下的所有 train_updates.txt (支持指定父目录包含多个seed的情况)
                    for root, dirs, files in os.walk(p):
                        if "train_updates.txt" in files:
                            file_path = os.path.join(root, "train_updates.txt")
                            df = parse_train_updates(file_path)
                            if not df.empty:
                                df['seed'] = root
                                df_list.append(df)
        
        # 统一重置 t 字段：不依赖文件中的 t，而是根据索引顺序生成 (每行代表 1000 steps)
        # 支持降采样 (downsample)
        processed_df_list = []
        for df in df_list:
            if downsample_factor > 1:
                # 保存 seed 列 (因为 mean() 会丢弃非数值列)
                seed_val = df['seed'].iloc[0] if 'seed' in df.columns else None
                
                # 按每 downsample_factor 行分组求平均
                # 这里的 df.index 假设是 0, 1, 2... 连续整数
                df = df.groupby(df.index // int(downsample_factor)).mean(numeric_only=True)
                
                # 恢复 seed
                if seed_val is not None:
                    df['seed'] = seed_val
            
            # 重新生成 t 字段
            # 原间隔是 1000，降采样后间隔变为 1000 * downsample_factor
            df['t'] = (df.index + 1) * 1000 * int(downsample_factor)
            processed_df_list.append(df)

        if processed_df_list:
            data_dict[alg_name] = pd.concat(processed_df_list, ignore_index=True)
        else:
            print(f"Warning: No data found for {alg_name} in provided paths.")
            data_dict[alg_name] = pd.DataFrame()
            
    return data_dict

def main():
    # 数据根目录 (请根据实际情况修改)
    DATA_ROOT = "../data" 
    
    # 降采样因子 (Downsample Factor)
    # 如果图上的点太密，可以增大此值。
    # 设为 1 表示不降采样（原样绘制，每点间隔1000）。
    # 设为 2 表示每 2 个点取平均，间隔变为 2000。
    # 设为 10 表示每 10 个点取平均，间隔变为 10000。
    PLOT_DOWNSAMPLE_FACTOR = 10

    # === 实验配置区域 (请在此处手动修改) ===
    # 格式: "图例名称": ["路径1", "路径2"...] 或 "单个路径"
    # 路径可以是具体实验的 seed 目录，也可以是包含多个 seed 的父目录
    EXPERIMENT_CONFIG = {
        'WCSAC (α=0.1)': [
            "data/2026-02-26_wcsac-0.1/2026-02-26_03-42-25-wcsac-0.1_s0"
        ],
        'WCSAC (α=0.5)': [
            "data/2026-02-24_wcsac-0.5/2026-02-24_14-16-39-wcsac-0.5_s0"
        ],
        'WCSAC (α=0.9)': [
            "data/2026-02-26_wcsac-0.9/2026-02-26_16-45-55-wcsac-0.9_s0"
        ],
        'SAC-Lagrangian': [
            "data/2026-02-20_saclag/2026-02-20_02-30-27-saclag_s0"
        ],
        'Standard-SAC': [
        ],
        'Fixed-Price': [
            "data/2026-02-26_fixed_price_baseline/2026-02-26_03-20-27-fixed_price_baseline_s0"
        ]
    }
    # ====================================

    # 加载数据
    data_dict = load_data_from_config(EXPERIMENT_CONFIG, downsample_factor=PLOT_DOWNSAMPLE_FACTOR)
    
    # 统一数据预处理：将 t 转换为 numeric 并除以 1e6
    for name, df in data_dict.items():
        if not df.empty:
            df['t'] = pd.to_numeric(df['t'])
            df['t'] = df['t'] / 1e6

    # 1. 绘制累积收益曲线 (WinRewMean)
    if any(not df.empty for df in data_dict.values()):
        plot_performance_curve(
            data_dict, 
            y_col='WinRewMean', 
            ylabel='Average Reward', 
            title='Training Performance: Average Reward',
            save_path='graph/graphs/reward_curve.png'
        )
    else:
        print("No data available to plot Reward Curve.")
    
    # 2. 绘制平均抢占率曲线 (WinCostMean)
    # 基于 WinCostMean 字段
    if any(not df.empty for df in data_dict.values()):
        plot_performance_curve(
            data_dict,
            y_col='WinCostMean',
            ylabel='Average Preemption Rate',
            title='Training Performance: Preemption Rate',
            save_path='graph/graphs/preemption_rate_curve.png'
        )
    else:
        print("No data available to plot Preemption Rate Curve.")

    # 3. 绘制 CVaR 变化曲线 (仅 WCSAC)
    plot_cvar_curve(
        data_dict,
        save_path='graph/graphs/cvar_cost_comparison.png'
    )
    
    # 4. 生成合并的对比图 (1行3列)
    print("Generating combined plot...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 6)) # 增加宽度，适应3个图
    
    # 子图1：Reward
    plot_performance_curve(
        data_dict, 
        y_col='WinRewMean', 
        ylabel='Average Reward', 
        title='(a) Average Reward', 
        ax=axes[0]
    )
    
    # 子图2：CVaR / Risk
    plot_cvar_curve(
        data_dict, 
        ax=axes[1]
    )
    axes[1].set_title('(b) Risk (CVaR) vs Expected Cost') # 覆盖默认标题
    
    # 子图3：Preemption Rate
    plot_performance_curve(
        data_dict,
        y_col='WinCostMean',
        ylabel='Average Preemption Rate',
        title='(c) Preemption Rate',
        ax=axes[2]
    )
    
    plt.tight_layout()
    plt.savefig('graph/graphs/combined_results.png', dpi=300, bbox_inches='tight')
    print("Saved combined plot to combined_results.png")

if __name__ == "__main__":
    main()
