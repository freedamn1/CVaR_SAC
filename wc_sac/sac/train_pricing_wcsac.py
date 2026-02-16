"""
动态定价任务的 WCSAC 训练入口（不依赖 Safety Gym）。

流程：
1) 先用 `wc_sac/dataProcess/getData.py` 把 traces 预处理成 excessive_capacity_cpu.npz
2) 再运行本脚本，用 excessive_capacity_cpu 驱动环境状态 C_t

示例：
python -m wc_sac.sac.train_pricing_wcsac --excessive_capacity_npz wc_sac/dataset/excessive_capacity_cpu.npz --p_min 0.1 --p_max 1.0 --dt 300 --horizon 288 --cost_lim 10
"""

from __future__ import annotations

import argparse

import numpy as np

from wc_sac.envs.preemptive_pricing_env import (
    PreemptivePricingEnv,
    PricingEnvConfig,
    ExcessiveCapacitySeries,
)
from wc_sac.sac.wcsac import sac
from wc_sac.utils.run_utils import setup_logger_kwargs


def make_poisson_rates(eta: float, thet: float, k: float, omega: float):
    # 泊松率函数：
    # f(p) = eta * (1 - p**k) ** omega
    # g(p) = thet - thet * (1 - p**k) ** omega
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excessive_capacity_npz", type=str, default="wc_sac/dataset/excessive_capacity_cpu.npz")
    parser.add_argument("--p_min", type=float, default=0.01)
    parser.add_argument("--p_max", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=300.0)
    parser.add_argument("--horizon", type=int, default=288)
    parser.add_argument("--n0", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    # 泊松率函数参数：f(p)=eta*(1-p**k)**omega, g(p)=thet-thet*(1-p**k)**omega
    parser.add_argument("--eta", type=float, default=0.33, help="arrival scale eta")
    parser.add_argument("--thet", type=float, default=0.33, help="leave scale thet")
    parser.add_argument("--k", type=float, default=2.0, help="exponent k on p")
    parser.add_argument("--omega", type=float, default=2.4, help="power omega on (1-p**k)")

    # 训练超参：保持与原 wcsac.py 一致的命名
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha_sig_level", type=float, default=0.5, help="CVaR significance level (tail proportion), e.g., 0.1 for worst 10%%")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--exp_name", type=str, default="pricing_wcsac")
    parser.add_argument("--steps_per_epoch", type=int, default=30000)
    parser.add_argument("--update_freq", type=int, default=100)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--local_start_steps", default=500, type=int)
    parser.add_argument("--local_update_after", default=500, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--fixed_entropy_bonus", default=None, type=float)
    parser.add_argument("--entropy_constraint", type=float, default=-1)
    parser.add_argument("--fixed_cost_penalty", default=None, type=float)
    parser.add_argument("--cost_lim", type=float, default=3.0)
    parser.add_argument("--zeta", type=float, default=0.1, help="tolerance ratio in [0,1): constrain cost CVaR into (cost_lim*(1-zeta), cost_lim)")
    parser.add_argument("--lr_s", type=int, default=0.1)
    parser.add_argument("--damp_s", type=int, default=10)
    parser.add_argument("--reward_scale", type=float, default=1e-3)
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="从指定训练目录恢复参数继续训练（目录内含 checkpoints/ 或 simple_save*）",
    )
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

    sac(
        env_fn,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        alpha_sig_level=args.alpha_sig_level,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        update_freq=args.update_freq,
        lr=args.lr,
        local_start_steps=args.local_start_steps,
        local_update_after=args.local_update_after,
        fixed_entropy_bonus=args.fixed_entropy_bonus,
        entropy_constraint=args.entropy_constraint,
        fixed_cost_penalty=args.fixed_cost_penalty,
        cost_lim=args.cost_lim,
        zeta=args.zeta,
        lr_scale=args.lr_s,
        damp_scale=args.damp_s,
        reward_scale=args.reward_scale,
        max_ep_len=args.horizon,
        logger_kwargs=logger_kwargs,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()

