from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from wc_sac.envs.preemptive_pricing_env import ExcessiveCapacitySeries, PreemptivePricingEnv, PricingEnvConfig

PRICES: list[float] = [
    0.1,
    0.2,
    0.3,
    0.4,
]

LABELS: list[str] = [
    "p=0.1",
    "p=0.2",
    "p=0.3",
    "p=0.4",
]


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


def price_to_action(price: float, p_min: float, p_max: float) -> float:
    p_min = float(p_min)
    p_max = float(p_max)
    price = float(price)
    if p_max <= p_min:
        return 0.0
    a_norm = 2.0 * (price - p_min) / (p_max - p_min) - 1.0
    return float(np.clip(a_norm, -1.0, 1.0))


def rollout_fixed_traces(
    env: PreemptivePricingEnv,
    price: float,
    episodes: int,
    print_freq: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, int]:
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    ret_traces = []
    cost_traces = []
    for ep in range(episodes):
        o = env.reset_for_test()
        done = False
        ep_len = 0
        cum_ret = 0.0
        ret_trace = []
        cost_trace = []
        while not done:
            a_norm = price_to_action(price, env.cfg.p_min, env.cfg.p_max)
            a = np.asarray([a_norm], dtype=np.float32)
            o, r, done, info = env.step(a)
            cum_ret += float(r)
            ret_trace.append(float(cum_ret))
            cost = float(info.get("cost", 0.0))
            cost_trace.append(cost)
            ep_len += 1
            if print_freq > 0 and (ep_len % int(print_freq) == 0 or done):
                price_info = float(info.get("price", np.nan))
                preempted = float(info.get("preempted", np.nan))
                n_running = float(info.get("n_running", np.nan))
                excessive_capacity = float(info.get("excessive_capacity", np.nan))
                print(
                    f"[rollout] ep={ep:3d} t={ep_len:4d} price={price_info: .3f} cost={cost: .6f} "
                    f"preempted={preempted: .3f} n_running={n_running: .3f} cap={excessive_capacity: .3f} "
                    f"cum_ret={cum_ret: .3f}"
                )
        ret_traces.append(np.asarray(ret_trace, dtype=np.float64))
        cost_traces.append(np.asarray(cost_trace, dtype=np.float64))

    lens = {int(x.size) for x in ret_traces}
    if len(lens) != 1:
        raise ValueError(f"Episode trace lengths are not equal: {sorted(lens)}")
    mean_ret = np.mean(np.stack(ret_traces, axis=0), axis=0)
    mean_cost = np.mean(np.stack(cost_traces, axis=0), axis=0)
    avg_cumret = float(mean_ret[-1]) if mean_ret.size else float("nan")
    avg_cost = float(np.mean(mean_cost)) if mean_cost.size else float("nan")
    over_ratio = float(np.mean(mean_cost > float(threshold))) if mean_cost.size else float("nan")
    return mean_ret, mean_cost, avg_cumret, avg_cost, over_ratio, int(mean_cost.size)


def _aggregate_trace(trace: np.ndarray, num_bins: int) -> tuple[np.ndarray, np.ndarray]:
    trace = np.asarray(trace, dtype=np.float64).reshape(-1)
    n = int(trace.size)
    if n <= 0:
        raise ValueError("Empty trace.")
    num_bins = int(num_bins)
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    num_bins = min(num_bins, 20, n)
    idx_chunks = np.array_split(np.arange(n, dtype=np.int32), num_bins)
    y = np.array([float(np.mean(trace[idx])) for idx in idx_chunks], dtype=np.float64)
    x = np.array([float(np.mean(idx)) for idx in idx_chunks], dtype=np.float64)
    return x, y


def save_multi_cumret_plot(
    traces: dict[str, np.ndarray],
    dt_seconds: float,
    num_bins: int,
    out_dir: Path,
    title: str,
    seed: int,
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError("缺少 matplotlib：请先安装 matplotlib") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"fixed_cumulative_return_s{int(seed)}.png"

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, trace in traces.items():
        steps_agg, y_agg = _aggregate_trace(trace, num_bins=num_bins)
        times = steps_agg * float(dt_seconds)
        ax.plot(times, y_agg, linewidth=1.6, label=label)

    ax.set_xlabel("time (seconds)")
    ax.set_ylabel("cumulative_return")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=170)
    plt.close(fig)
    return str(out_png)


def save_multi_preemption_plot(
    traces: dict[str, np.ndarray],
    dt_seconds: float,
    num_bins: int,
    out_dir: Path,
    title: str,
    threshold: float,
    seed: int,
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError("缺少 matplotlib：请先安装 matplotlib") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"fixed_preemption_rates_s{int(seed)}.png"

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, trace in traces.items():
        steps_agg, rate_agg = _aggregate_trace(trace, num_bins=num_bins)
        times = steps_agg * float(dt_seconds)
        ax.plot(times, rate_agg, linewidth=1.4, label=label)

    ax.axhline(float(threshold), color="black", linestyle="--", linewidth=1.2, label="threshold")
    ax.set_xlabel("time (seconds)")
    ax.set_ylabel("preemption_rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=170)
    plt.close(fig)
    return str(out_png)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1, help="每个策略测试 episode 数（取均值曲线）")
    parser.add_argument("--print_freq", type=int, default=0, help="每多少步打印一次 rollout 信息，0 表示不打印")
    parser.add_argument("--bins", type=int, default=20, help="绘图时对轨迹做分箱聚合；每条曲线最多保留 20 个点")
    parser.add_argument("--title_ret", type=str, default="Fixed Pricing Cumulative Return")
    parser.add_argument("--title_cost", type=str, default="Fixed Pricing Preemption Rate")
    parser.add_argument("--out_dir", type=str, default="graph/graphs", help="输出图片目录（相对 CVaR_SAC）")

    parser.add_argument("--excessive_capacity_npz", type=str, default="wc_sac/dataset/excessive_capacity_cpu_300sec.npz")
    parser.add_argument("--p_min", type=float, default=0.01)
    parser.add_argument("--p_max", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=300.0)
    parser.add_argument("--n0", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eta", type=float, default=0.33)
    parser.add_argument("--thet", type=float, default=0.33)
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--omega", type=float, default=2.4)
    parser.add_argument("--threshold", type=float, default=0.05)

    args = parser.parse_args(argv)

    prices = [float(x) for x in PRICES]
    if not prices:
        raise ValueError("PRICES 为空：请在脚本顶部配置要测试的定价列表。")

    labels = list(LABELS) if LABELS else [f"p={p:.3f}" for p in prices]
    if len(labels) != len(prices):
        raise ValueError(f"LABELS 长度必须与 PRICES 相同：len(LABELS)={len(labels)}, len(PRICES)={len(prices)}")

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

    ret_traces: dict[str, np.ndarray] = {}
    cost_traces: dict[str, np.ndarray] = {}
    avg_cumrets: dict[str, float] = {}
    avg_costs: dict[str, float] = {}
    over_ratios: dict[str, float] = {}
    step_counts: dict[str, int] = {}

    for price, label in zip(prices, labels):
        env = PreemptivePricingEnv(series, cfg, f_arrival_rate=f, g_departure_rate=g)
        ret_trace, cost_trace, avg_cumret, avg_cost, over_ratio, n_steps = rollout_fixed_traces(
            env=env,
            price=float(price),
            episodes=int(args.episodes),
            print_freq=int(args.print_freq),
            threshold=float(args.threshold),
        )
        ret_traces[str(label)] = ret_trace
        cost_traces[str(label)] = cost_trace
        avg_cumrets[str(label)] = float(avg_cumret)
        avg_costs[str(label)] = float(avg_cost)
        over_ratios[str(label)] = float(over_ratio)
        step_counts[str(label)] = int(n_steps)

    base_dir = Path(__file__).resolve().parent.parent
    out_dir = (base_dir / str(args.out_dir)).resolve()
    out_ret = save_multi_cumret_plot(
        traces=ret_traces,
        dt_seconds=float(cfg.dt),
        num_bins=int(args.bins),
        out_dir=out_dir,
        title=str(args.title_ret),
        seed=int(args.seed),
    )
    out_cost = save_multi_preemption_plot(
        traces=cost_traces,
        dt_seconds=float(cfg.dt),
        num_bins=int(args.bins),
        out_dir=out_dir,
        title=str(args.title_cost),
        threshold=float(args.threshold),
        seed=int(args.seed),
    )
    print(f"[summary] fixed pricing (threshold={float(args.threshold):.4f})")
    for label in ret_traces.keys():
        avg_ret = avg_cumrets.get(label, float("nan"))
        avg_cost = avg_costs.get(label, float("nan"))
        over_ratio = over_ratios.get(label, float("nan"))
        steps = step_counts.get(label, 0)
        print(f"  {label}: cum_ret={avg_ret:.6f}, preempt_avg={avg_cost:.6f}, over_ratio={over_ratio:.6f}, n={steps}")
    print(f"[ok] saved figure: {out_ret}")
    print(f"[ok] saved figure: {out_cost}")


if __name__ == "__main__":
    main()
