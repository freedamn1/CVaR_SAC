from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from wc_sac.envs.preemptive_pricing_env import ExcessiveCapacitySeries, PreemptivePricingEnv, PricingEnvConfig
from wc_sac.utils.load_utils import load_policy

###############################################################################
# Configure runs here (no CLI args needed)
###############################################################################
# Each entry should be a training output directory containing `simple_save*`.
FPATHS: list[str] = [
    "data/2026-02-20_saclag/2026-02-20_02-30-27-saclag_s0",
    "data/2026-02-26_wcsac-0.1/2026-02-26_03-42-25-wcsac-0.1_s0",
    "data/2026-02-24_wcsac-0.5/2026-02-24_23-38-56-wcsac-0.5_s0",
    "data/2026-02-26_wcsac-0.9/2026-02-26_16-45-55-wcsac-0.9_s0"
    # "data/2026-03-04_azure_saclag/2026-03-04_07-44-51-azure_saclag_s0",
    # "data/2026-03-05_azure_wcsac-0.5/2026-03-05_01-50-41-azure_wcsac-0.5_s1"
]

# Optional labels for plotting (must match length of FPATHS). Leave empty to
# auto-use directory names.
LABELS: list[str] = [
    "saclag",
    "wcsac-0.1",
    "wcsac-0.5",
    "wcsac-0.9"
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


def rollout_cumulative_return_trace(
    env: PreemptivePricingEnv,
    get_action: Callable[[np.ndarray], np.ndarray],
    episodes: int,
    print_freq: int,
) -> np.ndarray:
    """Run multiple test episodes and return mean cumulative-return trace (shape: [T])."""
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    traces = []
    for ep in range(episodes):
        o = env.reset_for_test()
        done = False
        ep_len = 0
        cum_ret = 0.0
        trace = []
        while not done:
            a = get_action(o)
            o, r, done, info = env.step(a)
            cum_ret += float(r)
            trace.append(float(cum_ret))
            ep_len += 1
            if print_freq > 0 and (ep_len % int(print_freq) == 0 or done):
                price = float(info.get("price", np.nan))
                cost = float(info.get("cost", np.nan))
                preempted = float(info.get("preempted", np.nan))
                n_running = float(info.get("n_running", np.nan))
                excessive_capacity = float(info.get("excessive_capacity", np.nan))
                print(
                    f"[rollout] ep={ep:3d} t={ep_len:4d} price={price: .3f} cost={cost: .6f} "
                    f"preempted={preempted: .3f} n_running={n_running: .3f} cap={excessive_capacity: .3f} "
                    f"cum_ret={cum_ret: .3f}"
                )
        traces.append(np.asarray(trace, dtype=np.float64))

    lens = {int(x.size) for x in traces}
    if len(lens) != 1:
        raise ValueError(f"Episode trace lengths are not equal: {sorted(lens)}")
    return np.mean(np.stack(traces, axis=0), axis=0)


def _aggregate_trace(trace: np.ndarray, num_bins: int) -> tuple[np.ndarray, np.ndarray]:
    trace = np.asarray(trace, dtype=np.float64).reshape(-1)
    n = int(trace.size)
    if n <= 0:
        raise ValueError("Empty trace.")
    num_bins = int(num_bins)
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    # Keep the same behavior as `wc_sac/sac/test_pricing_wcsac.py`:
    # aggregate to at most 20 points for readability.
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
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError("缺少 matplotlib：请先安装 matplotlib") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"cumulative_return.png"

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

    '''
    # 使用对数变换的版本（注释掉）
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, trace in traces.items():
        steps_agg, y_agg = _aggregate_trace(trace, num_bins=num_bins)
        # Use log-scale transform for better visual separation across runs.
        # Cumulative return is expected to be non-negative in this env, but we
        # still clip to be safe.
        y_agg = np.log1p(np.maximum(y_agg, 0.0))
        times = steps_agg * float(dt_seconds)
        ax.plot(times, y_agg, linewidth=1.6, label=label)

    ax.set_xlabel("time (seconds)")
    ax.set_ylabel("log1p(cumulative_return)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=170)
    plt.close(fig)
    return str(out_png)
    '''


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--itr", type=str, default="last", help="要加载的保存迭代：last 或整数（对所有运行生效）")
    parser.add_argument("--deterministic", action="store_true", help="用 mu（确定性）动作，否则用 pi")
    parser.add_argument("--episodes", type=int, default=1, help="每个模型测试 episode 数（累计收益曲线取均值）")
    parser.add_argument("--print_freq", type=int, default=0, help="每多少步打印一次 rollout 信息，0 表示不打印")
    parser.add_argument("--bins", type=int, default=20, help="绘图时对轨迹做分箱聚合；每条曲线最多保留 20 个点")
    parser.add_argument("--title", type=str, default="Cumulative Return (multi runs)")
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

    args = parser.parse_args(argv)

    fpaths = [str(x) for x in FPATHS]
    if not fpaths:
        raise ValueError("FPATHS 为空：请在脚本顶部配置要测试的训练输出目录列表。")

    labels = list(LABELS) if LABELS else [Path(p).name for p in fpaths]
    if len(labels) != len(fpaths):
        raise ValueError(f"LABELS 长度必须与 FPATHS 相同：len(LABELS)={len(labels)}, len(FPATHS)={len(fpaths)}")

    # Shared env config across all runs
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

    traces: dict[str, np.ndarray] = {}
    for fpath, label in zip(fpaths, labels):
        _env_from_ckpt, get_action, sess = load_policy(
            fpath=fpath, itr=args.itr, deterministic=bool(args.deterministic)
        )
        env = PreemptivePricingEnv(series, cfg, f_arrival_rate=f, g_departure_rate=g)
        try:
            trace = rollout_cumulative_return_trace(
                env=env,
                get_action=get_action,
                episodes=int(args.episodes),
                print_freq=int(args.print_freq),
            )
            traces[str(label)] = trace
        finally:
            sess.close()

    base_dir = Path(__file__).resolve().parent.parent
    out_dir = (base_dir / str(args.out_dir)).resolve()
    out_png = save_multi_cumret_plot(
        traces=traces,
        dt_seconds=float(cfg.dt),
        num_bins=int(args.bins),
        out_dir=out_dir,
        title=str(args.title),
    )
    print(f"[ok] saved figure: {out_png}")


if __name__ == "__main__":
    main()

