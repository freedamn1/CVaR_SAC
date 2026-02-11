"""
baselineCVaR.py

用途：
- 在动态定价环境（Poisson 到达/离开采样）中评估一个固定价格基线策略的成本分布；
- 通过采样多条 episode 估计上尾 CVaR（以及均值/分位数等统计量）；
- 可选导出单条 episode 的“即时成本 & 过剩容量”随时间步变化曲线（双 y 轴）。

运行示例：
python -m wc_sac.sac.baselineCVaR --price 0.5 --episodes 1000 --plot_out_png baseline_cost_capacity.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from wc_sac.envs.preemptive_pricing_env import ExcessiveCapacitySeries, PreemptivePricingEnv, PricingEnvConfig


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


@dataclass(frozen=True)
class BaselineResult:
    gamma: float
    alpha_tail: float
    price: float
    episodes: int
    discounted_costs: np.ndarray
    undiscounted_costs: np.ndarray
    trace_costs: Optional[np.ndarray]
    trace_excessive_capacity: Optional[np.ndarray]


@dataclass(frozen=True)
class EpisodeStats:
    discounted_cost: float
    undiscounted_cost: float
    total_reward: float
    count_cost_gt_0: int
    count_cost_gt_0_1: int
    count_cost_gt_0_2: int
    count_cost_gt_0_3: int
    count_cost_gt_0_4: int
    count_cost_gt_0_5: int


def _price_to_action_norm(price: float, p_min: float, p_max: float) -> float:
    p_min = float(p_min)
    p_max = float(p_max)
    if p_max <= p_min:
        raise ValueError(f"Invalid price range: p_min={p_min}, p_max={p_max}")
    p = float(np.clip(float(price), p_min, p_max))
    a = 2.0 * (p - p_min) / (p_max - p_min) - 1.0
    return float(np.clip(a, -1.0, 1.0))


def _rollout_episode_stats(
    env: PreemptivePricingEnv,
    fixed_price: float,
    gamma: float,
) -> EpisodeStats:
    obs = env.reset()
    _ = obs
    discounted_cost = 0.0
    undiscounted_cost = 0.0
    total_reward = 0.0
    disc = 1.0
    done = False
    t = 0

    count_cost_gt_0 = 0
    count_cost_gt_0_1 = 0
    count_cost_gt_0_2 = 0
    count_cost_gt_0_3 = 0
    count_cost_gt_0_4 = 0
    count_cost_gt_0_5 = 0

    a_norm = _price_to_action_norm(
        price=float(fixed_price), p_min=float(env.cfg.p_min), p_max=float(env.cfg.p_max)
    )
    while not done:
        action = np.array([a_norm], dtype=np.float32)
        obs, r, done, info = env.step(action)
        _ = (obs, info)
        c = float(info.get("cost", 0.0))

        total_reward += float(r)
        undiscounted_cost += c
        discounted_cost += disc * c
        disc *= float(gamma)

        if c > 0.0:
            count_cost_gt_0 += 1
        if c > 0.1:
            count_cost_gt_0_1 += 1
        if c > 0.2:
            count_cost_gt_0_2 += 1
        if c > 0.3:
            count_cost_gt_0_3 += 1
        if c > 0.4:
            count_cost_gt_0_4 += 1
        if c > 0.5:
            count_cost_gt_0_5 += 1

        t += 1
        if t > 10_000:
            raise RuntimeError("Episode did not terminate within 10000 steps.")

    return EpisodeStats(
        discounted_cost=float(discounted_cost),
        undiscounted_cost=float(undiscounted_cost),
        total_reward=float(total_reward),
        count_cost_gt_0=int(count_cost_gt_0),
        count_cost_gt_0_1=int(count_cost_gt_0_1),
        count_cost_gt_0_2=int(count_cost_gt_0_2),
        count_cost_gt_0_3=int(count_cost_gt_0_3),
        count_cost_gt_0_4=int(count_cost_gt_0_4),
        count_cost_gt_0_5=int(count_cost_gt_0_5),
    )


def _rollout_one_episode(
    env: PreemptivePricingEnv,
    fixed_price: float,
    gamma: float,
) -> Tuple[float, float]:
    obs = env.reset()
    _ = obs
    discounted = 0.0
    undiscounted = 0.0
    disc = 1.0
    done = False
    t = 0
    a_norm = _price_to_action_norm(
        price=float(fixed_price), p_min=float(env.cfg.p_min), p_max=float(env.cfg.p_max)
    )
    while not done:
        action = np.array([a_norm], dtype=np.float32)
        obs, r, done, info = env.step(action)
        _ = (obs, r)
        c = float(info.get("cost", 0.0))
        undiscounted += c
        discounted += disc * c
        disc *= float(gamma)
        t += 1
        if t > 10_000:
            raise RuntimeError("Episode did not terminate within 10000 steps.")
    return discounted, undiscounted


def _rollout_one_episode_with_trace(
    env: PreemptivePricingEnv,
    fixed_price: float,
    gamma: float,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    obs = env.reset()
    _ = obs
    discounted = 0.0
    undiscounted = 0.0
    disc = 1.0
    done = False
    t = 0
    costs: List[float] = []
    capacities: List[float] = []
    a_norm = _price_to_action_norm(
        price=float(fixed_price), p_min=float(env.cfg.p_min), p_max=float(env.cfg.p_max)
    )
    while not done:
        action = np.array([a_norm], dtype=np.float32)
        obs, r, done, info = env.step(action)
        _ = (obs, r)
        c = float(info.get("cost", 0.0))
        cap = float(info.get("excessive_capacity", 0.0))
        costs.append(c)
        capacities.append(cap)
        undiscounted += c
        discounted += disc * c
        disc *= float(gamma)
        t += 1
        if t > 10_000:
            raise RuntimeError("Episode did not terminate within 10000 steps.")
    return (
        discounted,
        undiscounted,
        np.asarray(costs, dtype=np.float64),
        np.asarray(capacities, dtype=np.float64),
    )


def _cvar_upper_tail(x: np.ndarray, alpha_tail: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("Empty array for CVaR.")
    if not (0.0 < float(alpha_tail) <= 1.0):
        raise ValueError(f"alpha_tail must be in (0,1], got {alpha_tail}.")
    k = int(np.ceil(float(alpha_tail) * x.size))
    k = max(1, min(k, x.size))
    x_sorted = np.sort(x)
    tail = x_sorted[-k:]
    return float(np.mean(tail))


def run_baseline_cvar(
    excessive_capacity_npz: str | Path,
    p_min: float,
    p_max: float,
    dt: float,
    horizon: int,
    n0: float,
    eta: float,
    thet: float,
    k: float,
    omega: float,
    fixed_price: float = 0.5,
    episodes: int = 1000,
    gamma: float = 0.99,
    alpha_tail: float = 0.1,
    seed: int = 0,
    trace_episode: int = 0,
) -> BaselineResult:
    path = Path(excessive_capacity_npz)
    data = np.load(str(path), allow_pickle=True)
    if "excessive_capacity_cpu" not in data:
        raise KeyError("NPZ file must contain 'excessive_capacity_cpu' key.")
    raw_capacity = np.asarray(data["excessive_capacity_cpu"], dtype=np.float32).reshape(-1)
    if raw_capacity.size < 2:
        raise ValueError("excessive_capacity_cpu length must be >= 2.")

    fixed_price = float(np.clip(fixed_price, float(p_min), float(p_max)))
    f, g = make_poisson_rates(float(eta), float(thet), float(k), float(omega))

    discounted_costs: List[float] = []
    undiscounted_costs: List[float] = []
    trace_costs: Optional[np.ndarray] = None
    trace_excessive_capacity: Optional[np.ndarray] = None
    for ep in range(int(episodes)):
        series = ExcessiveCapacitySeries(raw_capacity.copy())
        cfg = PricingEnvConfig(
            p_min=float(p_min),
            p_max=float(p_max),
            dt=float(dt),
            horizon=int(horizon),
            n0=float(n0),
            seed=int(seed) + int(ep),
        )
        env = PreemptivePricingEnv(series, cfg, f_arrival_rate=f, g_departure_rate=g)
        if int(ep) == int(trace_episode):
            disc_c, undis_c, c_seq, cap_seq = _rollout_one_episode_with_trace(
                env, fixed_price=fixed_price, gamma=float(gamma)
            )
            trace_costs = c_seq
            trace_excessive_capacity = cap_seq
        else:
            disc_c, undis_c = _rollout_one_episode(env, fixed_price=fixed_price, gamma=float(gamma))
        discounted_costs.append(disc_c)
        undiscounted_costs.append(undis_c)

    return BaselineResult(
        gamma=float(gamma),
        alpha_tail=float(alpha_tail),
        price=float(fixed_price),
        episodes=int(episodes),
        discounted_costs=np.asarray(discounted_costs, dtype=np.float64),
        undiscounted_costs=np.asarray(undiscounted_costs, dtype=np.float64),
        trace_costs=trace_costs,
        trace_excessive_capacity=trace_excessive_capacity,
    )


def _summarize(x: np.ndarray, alpha_tail: float) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
        "cvar_upper_tail": float(_cvar_upper_tail(x, alpha_tail=float(alpha_tail))),
    }


def _plot_cost_and_capacity(
    costs: np.ndarray,
    excessive_capacity: np.ndarray,
    out_png: str | Path,
    title: str,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError("缺少 matplotlib：请先 pip install matplotlib") from e

    costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    excessive_capacity = np.asarray(excessive_capacity, dtype=np.float64).reshape(-1)
    if costs.size == 0 or excessive_capacity.size == 0:
        raise ValueError("Empty trace for plotting.")
    if costs.size != excessive_capacity.size:
        raise ValueError(
            f"Trace length mismatch: costs={costs.size}, excessive_capacity={excessive_capacity.size}"
        )

    steps = np.arange(costs.size, dtype=np.int32)
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(steps, costs, linewidth=1.0, color="#d62728")
    ax1.set_xlabel("time step")
    ax1.set_ylabel("instant_cost (preemption rate)", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")

    ax2 = ax1.twinx()
    ax2.plot(steps, excessive_capacity, linewidth=1.0, color="#1f77b4", alpha=0.8)
    ax2.set_ylabel("excessive_capacity", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    ax1.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=160)
    plt.close(fig)


def baseline_log_output(
    *,
    excessive_capacity_npz: str | Path,
    out_path: str | Path,
    p_min: float,
    p_max: float,
    dt: float,
    horizon: int,
    n0: float,
    eta: float,
    thet: float,
    k: float,
    omega: float,
    episodes: int,
    gamma: float,
    alpha_tail: float,
    seed: int,
) -> Path:
    path = Path(excessive_capacity_npz)
    data = np.load(str(path), allow_pickle=True)
    if "excessive_capacity_cpu" not in data:
        raise KeyError("NPZ file must contain 'excessive_capacity_cpu' key.")
    raw_capacity = np.asarray(data["excessive_capacity_cpu"], dtype=np.float32).reshape(-1)
    if raw_capacity.size < 2:
        raise ValueError("excessive_capacity_cpu length must be >= 2.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    f, g = make_poisson_rates(float(eta), float(thet), float(k), float(omega))
    price_grid = np.round(np.arange(0.1, 1.01, 0.1), 2).tolist()

    dataset_steps = int(raw_capacity.size)
    horizon = int(dataset_steps)
    dataset_max_episode_steps = int(min(int(horizon), dataset_steps - 1))

    with out_path.open("w", encoding="utf-8") as fw:
        fw.write("baseline_cost_allprice\n")
        fw.write(f"dataset_steps={dataset_steps} max_episode_steps={dataset_max_episode_steps}\n")
        fw.write(
            "grid_prices=0.1..1.0 step=0.1 "
            f"episodes={int(episodes)} gamma={float(gamma)} alpha_tail={float(alpha_tail)} "
            f"dt={float(dt)} horizon={int(horizon)} n0={float(n0)} seed={int(seed)}\n"
        )
        fw.write(f"poisson_params eta={float(eta)} thet={float(thet)} k={float(k)} omega={float(omega)}\n")
        fw.write("\n")

        for price in price_grid:
            p = float(np.clip(float(price), float(p_min), float(p_max)))

            discounted_costs: List[float] = []
            undiscounted_costs: List[float] = []
            total_rewards: List[float] = []

            count_cost_gt_0 = 0
            count_cost_gt_0_1 = 0
            count_cost_gt_0_2 = 0
            count_cost_gt_0_3 = 0
            count_cost_gt_0_4 = 0
            count_cost_gt_0_5 = 0

            for ep in range(int(episodes)):
                series = ExcessiveCapacitySeries(raw_capacity.copy())
                cfg = PricingEnvConfig(
                    p_min=float(p_min),
                    p_max=float(p_max),
                    dt=float(dt),
                    horizon=int(horizon),
                    n0=float(n0),
                    seed=int(seed) + int(ep),
                )
                env = PreemptivePricingEnv(series, cfg, f_arrival_rate=f, g_departure_rate=g)
                st = _rollout_episode_stats(env, fixed_price=p, gamma=float(gamma))
                discounted_costs.append(st.discounted_cost)
                undiscounted_costs.append(st.undiscounted_cost)
                total_rewards.append(st.total_reward)
                count_cost_gt_0 += st.count_cost_gt_0
                count_cost_gt_0_1 += st.count_cost_gt_0_1
                count_cost_gt_0_2 += st.count_cost_gt_0_2
                count_cost_gt_0_3 += st.count_cost_gt_0_3
                count_cost_gt_0_4 += st.count_cost_gt_0_4
                count_cost_gt_0_5 += st.count_cost_gt_0_5

            disc_stats = _summarize(np.asarray(discounted_costs, dtype=np.float64), alpha_tail=float(alpha_tail))
            undis_stats = _summarize(np.asarray(undiscounted_costs, dtype=np.float64), alpha_tail=float(alpha_tail))
            rew_stats = _summarize(np.asarray(total_rewards, dtype=np.float64), alpha_tail=1.0)

            denom_eps = max(1, int(episodes))
            mean_cost_gt_0 = float(count_cost_gt_0) / float(denom_eps)
            mean_cost_gt_0_1 = float(count_cost_gt_0_1) / float(denom_eps)
            mean_cost_gt_0_2 = float(count_cost_gt_0_2) / float(denom_eps)
            mean_cost_gt_0_3 = float(count_cost_gt_0_3) / float(denom_eps)
            mean_cost_gt_0_4 = float(count_cost_gt_0_4) / float(denom_eps)
            mean_cost_gt_0_5 = float(count_cost_gt_0_5) / float(denom_eps)

            fw.write(f"price={p:.2f}\n")
            fw.write("discounted_cost\n")
            for kk, vv in disc_stats.items():
                fw.write(f"  {kk}={vv:.6f}\n")
            fw.write("undiscounted_cost\n")
            for kk, vv in undis_stats.items():
                fw.write(f"  {kk}={vv:.6f}\n")
            fw.write("total_reward\n")
            for kk, vv in rew_stats.items():
                fw.write(f"  {kk}={vv:.6f}\n")
            fw.write("instant_cost_counts_mean_per_episode\n")
            fw.write(f"  gt_0={mean_cost_gt_0:.6f}\n")
            fw.write(f"  gt_0_1={mean_cost_gt_0_1:.6f}\n")
            fw.write(f"  gt_0_2={mean_cost_gt_0_2:.6f}\n")
            fw.write(f"  gt_0_3={mean_cost_gt_0_3:.6f}\n")
            fw.write(f"  gt_0_4={mean_cost_gt_0_4:.6f}\n")
            fw.write(f"  gt_0_5={mean_cost_gt_0_5:.6f}\n")
            fw.write("\n")

    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excessive_capacity_npz", type=str, default="wc_sac/dataset/excessive_capacity_cpu_300sec.npz")
    parser.add_argument("--p_min", type=float, default=0.1)
    parser.add_argument("--p_max", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=300.0)
    parser.add_argument("--horizon", type=int, default=288)
    parser.add_argument("--n0", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--eta", type=float, default=0.33)
    parser.add_argument("--thet", type=float, default=0.33)
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--omega", type=float, default=2.4)

    parser.add_argument("--price", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha_tail", type=float, default=0.1)
    parser.add_argument("--trace_episode", type=int, default=0)
    parser.add_argument("--log_all_prices", action="store_true")
    parser.add_argument(
        "--plot_out_png",
        type=str,
        nargs="?",
        const="data/baseLineCVaR/baseline_cost_capacity.png",
        default=None,
    )
    args = parser.parse_args()

    if bool(args.log_all_prices):
        out_path = baseline_log_output(
            excessive_capacity_npz=args.excessive_capacity_npz,
            out_path="data/baseLineCVaR/baseline_cost_allprice",
            p_min=args.p_min,
            p_max=args.p_max,
            dt=args.dt,
            horizon=args.horizon,
            n0=args.n0,
            eta=args.eta,
            thet=args.thet,
            k=args.k,
            omega=args.omega,
            episodes=args.episodes,
            gamma=args.gamma,
            alpha_tail=args.alpha_tail,
            seed=args.seed,
        )
        print(f"[ok] saved log: {out_path}")
        return

    result = run_baseline_cvar(
        excessive_capacity_npz=args.excessive_capacity_npz,
        p_min=args.p_min,
        p_max=args.p_max,
        dt=args.dt,
        horizon=args.horizon,
        n0=args.n0,
        eta=args.eta,
        thet=args.thet,
        k=args.k,
        omega=args.omega,
        fixed_price=args.price,
        episodes=args.episodes,
        gamma=args.gamma,
        alpha_tail=args.alpha_tail,
        seed=args.seed,
        trace_episode=args.trace_episode,
    )

    disc_stats = _summarize(result.discounted_costs, alpha_tail=result.alpha_tail)
    undis_stats = _summarize(result.undiscounted_costs, alpha_tail=result.alpha_tail)

    print("baseline_fixed_price_cvar")
    print(f"price={result.price} episodes={result.episodes} gamma={result.gamma} alpha_tail={result.alpha_tail}")
    print("discounted_cost")
    for k, v in disc_stats.items():
        print(f"  {k}={v:.6f}")
    print("undiscounted_cost")
    for k, v in undis_stats.items():
        print(f"  {k}={v:.6f}")

    if args.plot_out_png is not None:
        if result.trace_costs is None or result.trace_excessive_capacity is None:
            raise RuntimeError("Missing trace data: trace_episode was not recorded.")
        title = (
            f"baseline trace | price={result.price} | "
            f"episode={int(args.trace_episode)} | horizon={int(args.horizon)}"
        )
        _plot_cost_and_capacity(
            costs=result.trace_costs,
            excessive_capacity=result.trace_excessive_capacity,
            out_png=args.plot_out_png,
            title=title,
        )
        print(f"[ok] saved figure: {args.plot_out_png}")


if __name__ == "__main__":
    main()
