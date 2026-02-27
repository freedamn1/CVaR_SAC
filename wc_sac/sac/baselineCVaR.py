"""
baselineCVaR.py

用途：
- 模拟静态定价策略（默认价格 0.5），生成与 train_pricing_wcsac 类似的 train_updates.txt 日志文件。
- 该日志文件包含 WinRewMean, WinCostMean 等字段，供 plot_results.py 绘图使用。
- 不再生成单独的图片，而是通过长时间运行模拟生成训练曲线数据。

运行示例：
python -m wc_sac.sac.baselineCVaR --price 0.5 --cpu 1 --epochs 100
"""

from __future__ import annotations

import argparse
import time
import os
import numpy as np

from wc_sac.envs.preemptive_pricing_env import (
    PreemptivePricingEnv,
    PricingEnvConfig,
    ExcessiveCapacitySeries,
)
from wc_sac.utils.run_utils import setup_logger_kwargs
from wc_sac.utils.mpi_tools import mpi_fork, proc_id, num_procs


def make_poisson_rates(eta: float, thet: float, k: float, omega: float):
    def f(p: float) -> float:
        try:
            val = eta * (1.0 - float(p) ** k) ** omega
        except Exception:
            val = 0.0
        return max(float(val), 0.0)

    def g(p: float) -> float:
        try:
            val = thet - thet * (1.0 - float(p) ** k) ** omega
        except Exception:
            val = 0.0
        return max(float(val), 0.0)

    return f, g


def run_static_baseline(
    env_fn,
    price: float = 0.5,
    seed: int = 0,
    steps_per_epoch: int = 1000,
    epochs: int = 100,
    update_freq: int = 100,
    logger_kwargs: dict = dict(),
    reward_scale: float = 1.0,
    train_print_freq: int = 10, # 每隔多少个 update_freq 打印一次日志
):
    """
    模拟静态定价过程，并生成 train_updates.txt
    """
    # Setup logging
    output_dir = logger_kwargs.get('output_dir')
    train_log_f = None
    if proc_id() == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        train_log_f = open(os.path.join(output_dir, 'train_updates.txt'), 'w', encoding='utf-8')
        print(f"Logging to {os.path.join(output_dir, 'train_updates.txt')}")

    # Seed
    seed += 10000 * proc_id()
    np.random.seed(seed)

    env = env_fn()
    
    # Calculate normalized action for the fixed price
    p_min = env.cfg.p_min
    p_max = env.cfg.p_max
    
    def price_to_action(p):
        p = np.clip(p, p_min, p_max)
        return 2.0 * (p - p_min) / (p_max - p_min) - 1.0

    fixed_action = np.array([price_to_action(price)], dtype=np.float32)
    
    # Buffers for window-averaged statistics
    current_window_raw_rews = []
    current_window_costs = []
    
    o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
    total_steps = steps_per_epoch * epochs
    
    start_time = time.time()
    
    # Main loop
    for t in range(total_steps):
        
        a = fixed_action
        o2, r, d, info = env.step(a)
        
        c = info.get('cost', 0.0)
        
        # Track for window logging
        current_window_raw_rews.append(r)
        current_window_costs.append(c)
        
        r *= reward_scale
        ep_ret += r
        ep_cost += c
        ep_len += 1
        
        o = o2
        
        # Handle done
        if d or (ep_len == env.cfg.horizon):
             o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
        
        # Logging logic matching wcsac.py
        # Log every update_freq steps (simulating the training update frequency)
        if (t + 1) % update_freq == 0:
            
            # Only root process writes to file
            if proc_id() == 0:
                # Check if we should print (train_print_freq)
                update_block = (t // update_freq)
                if update_block % train_print_freq == 0:
                    
                    # Calculate window means
                    if current_window_raw_rews:
                        win_rew_mean = float(np.mean(current_window_raw_rews))
                    else:
                        win_rew_mean = 0.0
                        
                    if current_window_costs:
                        win_cost_mean = float(np.mean(current_window_costs))
                        win_cost_max = float(np.max(current_window_costs))
                        win_cost_nz = float(np.mean(np.array(current_window_costs) > 0.0))
                    else:
                        win_cost_mean = 0.0
                        win_cost_max = 0.0
                        win_cost_nz = 0.0
                    
                    # Reset window buffers
                    current_window_raw_rews = []
                    current_window_costs = []
                    
                    # Construct message matching wcsac.py format
                    # Filling missing fields with 0.0 or appropriate defaults
                    # [train] t=... | LossPi=... ... | WinRewMean=... | WinCostMean=...
                    
                    # Note: For static policy, QcPi (Expected Cost) is essentially WinCostMean
                    qc_pi_val = win_cost_mean 
                    
                    msg = (
                        f"[train] t={t+1:6d} | LossPi=0.0000 "
                        f"| PiEntropy=0.0000 | Alpha=0.0000 "
                        f"| MinQ=0.0000 "
                        f"| QcPiCVaR={qc_pi_val: .4f} | QcPi={qc_pi_val: .4f} | QcPiVar=0.0000"
                        f"| BatchCostMean={win_cost_mean: .4f} | BatchCostMax={win_cost_max: .4f} | BatchCostNZ={win_cost_nz: .4f}"
                        f"| WinRewMean={win_rew_mean: .4f} | WinCostMean={win_cost_mean: .4f}"
                    )
                    
                    train_log_f.write(msg + "\n")
                    train_log_f.flush()

    if proc_id() == 0:
        if train_log_f:
            train_log_f.close()
        print(f"Finished. Logs saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excessive_capacity_npz", type=str, default="wc_sac/dataset/excessive_capacity_cpu_300sec.npz")
    parser.add_argument("--p_min", type=float, default=0.01)
    parser.add_argument("--p_max", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=300.0)
    parser.add_argument("--horizon", type=int, default=288)
    parser.add_argument("--n0", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    # 泊松率函数参数
    parser.add_argument("--eta", type=float, default=0.33)
    parser.add_argument("--thet", type=float, default=0.33)
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--omega", type=float, default=2.4)

    # 静态策略参数
    parser.add_argument("--price", type=float, default=0.5, help="Static fixed price")
    
    # 模拟参数 (保持与 train_pricing_wcsac 一致的接口)
    parser.add_argument("--exp_name", type=str, default="fixed_price_baseline")
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=30000)
    parser.add_argument("--update_freq", type=int, default=100)
    parser.add_argument("--train_print_freq", type=int, default=10) # 每10个update打印一次，即1000 steps
    parser.add_argument("--reward_scale", type=float, default=1e-3)

    args = parser.parse_args()

    series = ExcessiveCapacitySeries.from_npz(args.excessive_capacity_npz)
    cfg = PricingEnvConfig(
        p_min=args.p_min,
        p_max=args.p_max,
        dt=args.dt,
        horizon=args.horizon,
        n0=args.n0,
        seed=args.seed,
    )
    f, g = make_poisson_rates(args.eta, args.thet, args.k, args.omega)

    def env_fn():
        return PreemptivePricingEnv(series, cfg, f_arrival_rate=f, g_departure_rate=g)

    # 设置日志配置
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed=args.seed)

    run_static_baseline(
        env_fn,
        price=args.price,
        seed=args.seed,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        update_freq=args.update_freq,
        logger_kwargs=logger_kwargs,
        reward_scale=args.reward_scale,
        train_print_freq=args.train_print_freq
    )


if __name__ == "__main__":
    main()
