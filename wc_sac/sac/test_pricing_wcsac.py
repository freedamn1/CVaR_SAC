from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from wc_sac.envs.preemptive_pricing_env import ExcessiveCapacitySeries, PreemptivePricingEnv, PricingEnvConfig
from wc_sac.utils.load_utils import load_policy


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


def _save_preemption_plot(preemption_rates: np.ndarray, dt_seconds: float, ep: int) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError("缺少 matplotlib：请先安装 matplotlib") from e

    preemption_rates = np.asarray(preemption_rates, dtype=np.float64).reshape(-1)
    n = int(preemption_rates.size)
    if n == 0:
        raise ValueError("Empty preemption_rates.")

    num_bins = min(20, n)
    idx_chunks = np.array_split(np.arange(n, dtype=np.int32), num_bins)
    rates_agg = np.array([float(np.mean(preemption_rates[idx])) for idx in idx_chunks], dtype=np.float64)
    steps_agg = np.array([float(np.mean(idx)) for idx in idx_chunks], dtype=np.float64)
    times = steps_agg * float(dt_seconds)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, rates_agg, linewidth=1.2, color="#d62728")
    ax.set_xlabel("time (seconds)")
    ax.set_ylabel("preemption_rate")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / "dataset"
    if not out_dir.exists():
        raise FileNotFoundError(f"dataset 目录不存在：{out_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_png = out_dir / f"preemption_rate_in_test_{ts}_ep{int(ep)}.png"
    fig.savefig(str(out_png), dpi=160)
    plt.close(fig)
    return str(out_png)


def rollout(
    env: PreemptivePricingEnv,
    get_action: Callable[[np.ndarray], np.ndarray],
    episodes: int,
    deterministic: bool,
    print_freq: int,
) -> None:
    _ = deterministic
    for ep in range(int(episodes)):
        o = env.reset_for_test()
        done = False
        ep_ret = 0.0
        ep_cost = 0.0
        ep_len = 0
        preemption_rate_trace = []
        while not done:
            a = get_action(o)
            o, r, done, info = env.step(a)
            preemption_rate_trace.append(float(info.get("cost", 0.0)))
            ep_ret += float(r)
            ep_cost += float(info.get("cost", 0.0))
            ep_len += 1
            if print_freq > 0 and (ep_len % int(print_freq) == 0 or done):
                price = float(info.get("price", np.nan))
                cost = float(info.get("cost", np.nan))
                preempted = float(info.get("preempted", np.nan))
                n_running = float(info.get("n_running", np.nan))
                excessive_capacity = float(info.get("excessive_capacity", np.nan))
                print(
                    f"[rollout] ep={ep:4d} t={ep_len:4d} price={price: .3f} cost={cost: .6f} "
                    f"preempted={preempted: .3f} n_running={n_running: .3f} cap={excessive_capacity: .3f}"
                )
        print(f"[episode] ep={ep:4d} EpRet={ep_ret: .6f} EpCost={ep_cost: .6f} EpLen={ep_len}")
        out_png = _save_preemption_plot(np.asarray(preemption_rate_trace), dt_seconds=float(env.cfg.dt), ep=ep)
        print(f"[ok] saved figure: {out_png}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", type=str, default="data/2026-02-13_pricing_wcsac/2026-02-13_07-15-27-pricing_wcsac_s0", help="训练输出目录（包含 simple_save* 子目录）")
    parser.add_argument("--itr", type=str, default="last", help="要加载的保存迭代：last 或整数")
    parser.add_argument("--deterministic", action="store_true", help="用 mu（确定性）动作，否则用 pi")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--print_freq", type=int, default=10, help="每多少步打印一次 rollout 信息，0 表示不打印")

    parser.add_argument("--excessive_capacity_npz", type=str, default="wc_sac/dataset/excessive_capacity_cpu.npz")
    parser.add_argument("--p_min", type=float, default=0.01)
    parser.add_argument("--p_max", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=300.0)
    parser.add_argument("--n0", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eta", type=float, default=0.33)
    parser.add_argument("--thet", type=float, default=0.33)
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--omega", type=float, default=2.4)

    args = parser.parse_args(argv)

    _, get_action, sess = load_policy(fpath=args.fpath, itr=args.itr, deterministic=bool(args.deterministic))

    series = ExcessiveCapacitySeries.from_npz(args.excessive_capacity_npz)
    cfg = PricingEnvConfig(
        p_min=float(args.p_min),
        p_max=float(args.p_max),
        dt=float(args.dt),
        horizon=max(int(len(series) - 1), 1),
        n0=float(args.n0),
        seed=int(args.seed),
    )
    f, g = make_poisson_rates(float(args.eta), float(args.thet), float(args.k), float(args.omega))
    env = PreemptivePricingEnv(series, cfg, f_arrival_rate=f, g_departure_rate=g)

    try:
        rollout(
            env=env,
            get_action=get_action,
            episodes=int(args.episodes),
            deterministic=bool(args.deterministic),
            print_freq=int(args.print_freq),
        )
    finally:
        sess.close()


if __name__ == "__main__":
    main()
