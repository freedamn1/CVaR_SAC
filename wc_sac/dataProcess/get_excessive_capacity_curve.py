from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_excessive_capacity(
    npz_path: str | Path,
    out_png: str | Path | None = None,
    show: bool = True,
    max_points: int = 200_000,
) -> None:
    """
    画出过剩容量（excessive_capacity_cpu）随时间的变化曲线。

    - 横轴：times（npz 中的 times；如果没有则用 dt_seconds 生成）
    - 纵轴：excessive_capacity_cpu（单位：CPU core）
    """
    npz_path = Path(npz_path)
    data = np.load(str(npz_path), allow_pickle=True)

    if "excessive_capacity_cpu" not in data:
        raise KeyError("npz 缺少 key: 'excessive_capacity_cpu'")

    excessive = np.asarray(data["excessive_capacity_cpu"], dtype=np.float32)
    dt_seconds = int(np.asarray(data["dt_seconds"]).item()) if "dt_seconds" in data else None

    if "times" in data:
        times = np.asarray(data["times"], dtype=np.int64)
    else:
        if dt_seconds is None:
            raise KeyError("npz 缺少 'times' 且缺少 'dt_seconds'，无法生成时间轴。")
        times = (np.arange(len(excessive), dtype=np.int64) * dt_seconds).astype(np.int64)

    if len(times) != len(excessive):
        raise ValueError(f"times 长度({len(times)})与 excessive_capacity_cpu 长度({len(excessive)})不一致")

    # 点太多会很慢：均匀抽样
    n = len(excessive)
    if max_points is not None and n > int(max_points):
        idx = np.linspace(0, n - 1, int(max_points), dtype=np.int64)
        times_plot = times[idx]
        excessive_plot = excessive[idx]
    else:
        times_plot = times
        excessive_plot = excessive

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError("缺少 matplotlib：请先 pip install matplotlib") from e

    plt.figure(figsize=(12, 4))
    plt.plot(times_plot, excessive_plot, linewidth=0.8)
    plt.title(f"Excessive Capacity vs Time ({npz_path.name})")
    plt.xlabel("time (seconds)")
    plt.ylabel("excessive_capacity_cpu (core)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_png), dpi=160)

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # 默认画 300 秒序列；如果你想看 10 秒序列，把下面文件名改成 excessive_capacity_cpu_10sec.npz
    base_dir = Path(__file__).resolve().parent
    default_npz = base_dir / "excessive_capacity_cpu_300sec.npz"
    default_png = base_dir / "excessive_capacity_cpu_curve.png"

    print(f"[info] loading: {default_npz}")
    plot_excessive_capacity(default_npz, out_png=default_png, show=False, max_points=200_000)
    print(f"[ok] saved figure: {default_png}")
