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


def slice_excessive_capacity(npz_path: str, start_step: int, steps: int) -> np.ndarray:
    data = np.load(str(npz_path), allow_pickle=True)
    if "excessive_capacity_cpu" not in data:
        raise KeyError("NPZ file must contain 'excessive_capacity_cpu' key.")
    arr = np.asarray(data["excessive_capacity_cpu"], dtype=np.float32).reshape(-1)
    if arr.size < 2:
        raise ValueError("excessive_capacity_cpu 长度至少为 2。")

    start = int(start_step)
    n = int(steps)
    if start < 0:
        raise ValueError("start_step 不能小于 0。")
    if n <= 0:
        raise ValueError("steps 必须为正整数。")
    end = start + n + 1
    if end > arr.size:
        raise ValueError(
            f"截取越界：start_step={start}, steps={n}, 需要到 {end - 1}，但序列长度仅 {arr.size}。"
        )
    return arr[start:end].astype(np.float32)


def rollout_policy_window(
    env: PreemptivePricingEnv,
    get_action: Callable[[np.ndarray], np.ndarray],
    print_freq: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    o = env.reset_for_test()
    done = False
    t = 0
    excessive_trace = []
    occupied_trace = []
    price_trace = []
    while not done:
        a = get_action(o)
        o, r, done, info = env.step(a)
        excessive = float(info.get("excessive_capacity", np.nan))
        occupied = float(info.get("n_running", np.nan))
        price = float(info.get("price", np.nan))
        excessive_trace.append(excessive)
        occupied_trace.append(occupied)
        price_trace.append(price)
        t += 1
        if print_freq > 0 and (t % int(print_freq) == 0 or done):
            cost = float(info.get("cost", np.nan))
            preempted = float(info.get("preempted", np.nan))
            print(
                f"[rollout] t={t:4d} price={price: .3f} reward={float(r): .6f} cost={cost: .6f} "
                f"preempted={preempted: .3f} excessive={excessive: .3f} occupied={occupied: .3f}"
            )
    return (
        np.asarray(excessive_trace, dtype=np.float64),
        np.asarray(occupied_trace, dtype=np.float64),
        np.asarray(price_trace, dtype=np.float64),
    )


def save_policy_analysis_plot(
    excessive_trace: np.ndarray,
    occupied_trace: np.ndarray,
    price_trace: np.ndarray,
    start_step: int,
    out_dir: Path,
    seed: int,
    title: str,
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError("缺少 matplotlib：请先安装 matplotlib") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"pricing_policy_analysis_ali_wcsac_0.1_s{int(seed)}.png"

    x = np.arange(excessive_trace.size, dtype=np.int32) + int(start_step)
    fig, ax_left = plt.subplots(figsize=(14, 6))
    ax_right = ax_left.twinx()

    line_excessive, = ax_left.plot(
        x,
        excessive_trace,
        color="#1f77b4",
        linewidth=1.8,
        label="excessive_capacity",
    )
    line_occupied, = ax_left.plot(
        x,
        np.clip(occupied_trace, 0.0, 100.0),
        color="#2ca02c",
        linewidth=1.6,
        label="occupied_capacity",
    )
    bars = ax_right.bar(
        x,
        price_trace,
        width=0.8,
        color="#ff7f0e",
        alpha=0.25,
        label="price",
    )

    ax_left.set_xlabel("test step")
    ax_left.set_ylabel("occupied capacity (0~100)")
    ax_left.set_ylim(0.0, 100.0)
    ax_left.grid(True, alpha=0.25)

    ax_right.set_ylabel("price (0.1~1.0)")
    ax_right.set_ylim(0.1, 1.0)

    ax_left.set_title(title)
    handles = [line_excessive, line_occupied, bars]
    labels = [h.get_label() for h in handles]
    ax_left.legend(handles, labels, fontsize=9, loc="best")

    fig.tight_layout()
    fig.savefig(str(out_png), dpi=170)
    plt.close(fig)
    return str(out_png)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", type=str, default="data/2026-02-26_wcsac-0.1/2026-02-26_03-42-25-wcsac-0.1_s0")
    parser.add_argument("--itr", type=str, default="last")
    parser.add_argument("--deterministic", action="store_true", help="用 mu（确定性）动作，否则用 pi")
    parser.add_argument("--start_step", type=int, default=660, help="从原始序列的第几个时间步开始")
    parser.add_argument("--steps", type=int, default=288, help="分析窗口长度（时间步）")
    parser.add_argument("--print_freq", type=int, default=0, help="每多少步打印一次 rollout 信息，0 表示不打印")
    parser.add_argument("--title", type=str, default="policy_analysis_alibaba_wcsac_0.1")
    parser.add_argument("--out_dir", type=str, default="graph/graphs", help="输出图片目录（相对 CVaR_SAC）")

    parser.add_argument("--excessive_capacity_npz", type=str, default="wc_sac/dataset/excessive_capacity_cpu_300sec.npz")
    parser.add_argument("--p_min", type=float, default=0.1)
    parser.add_argument("--p_max", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=300.0)
    parser.add_argument("--n0", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eta", type=float, default=0.33)
    parser.add_argument("--thet", type=float, default=0.33)
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--omega", type=float, default=2.4)

    args = parser.parse_args(argv)

    segment = slice_excessive_capacity(
        npz_path=str(args.excessive_capacity_npz),
        start_step=int(args.start_step),
        steps=int(args.steps),
    )
    series = ExcessiveCapacitySeries(segment)

    _, get_action, sess = load_policy(
        fpath=str(args.fpath), itr=str(args.itr), deterministic=bool(args.deterministic)
    )
    cfg = PricingEnvConfig(
        p_min=float(args.p_min),
        p_max=float(args.p_max),
        dt=float(args.dt),
        horizon=int(args.steps),
        n0=float(args.n0),
        seed=int(args.seed),
    )
    f, g = make_poisson_rates(float(args.eta), float(args.thet), float(args.k), float(args.omega))
    env = PreemptivePricingEnv(series, cfg, f_arrival_rate=f, g_departure_rate=g)

    try:
        excessive_trace, occupied_trace, price_trace = rollout_policy_window(
            env=env,
            get_action=get_action,
            print_freq=int(args.print_freq),
        )
    finally:
        sess.close()

    base_dir = Path(__file__).resolve().parent.parent
    out_dir = (base_dir / str(args.out_dir)).resolve()
    out_png = save_policy_analysis_plot(
        excessive_trace=excessive_trace,
        occupied_trace=occupied_trace,
        price_trace=price_trace,
        start_step=int(args.start_step),
        out_dir=out_dir,
        seed=int(args.seed),
        title=str(args.title),
    )

    print("[summary] pricing policy window")
    print(
        f"  steps={int(excessive_trace.size)}, excessive_mean={float(np.mean(excessive_trace)):.6f}, "
        f"occupied_mean={float(np.mean(occupied_trace)):.6f}, price_mean={float(np.mean(price_trace)):.6f}"
    )
    print(f"[ok] saved figure: {out_png}")


if __name__ == "__main__":
    main()
