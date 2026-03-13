import numpy as np
from pathlib import Path
from wc_sac.envs.preemptive_pricing_env import (
    ExcessiveCapacitySeries,
    PreemptivePricingEnv,
    PricingEnvConfig,
)


def normalize_to_0_100(x: np.ndarray) -> np.ndarray:
    """把一维序列做 min-max 归一化到 [0, 100]。"""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    return ((x - x_min) / (x_max - x_min) * 100.0).astype(np.float32)
# npz keys: ['dt_seconds', 'capacity_cpu', 'times', 'usage_cpu', 'excessive_capacity_cpu']
# total capacityy = 387168.0
# 1) 加载 npz（从 dataset 目录）
base_dir = Path(__file__).resolve().parent.parent
path = base_dir / "dataset" / "simulate_workload_20260307_200316.npz"

# 详细检查 npz 内容
data = np.load(str(path), allow_pickle=True)

excessive_capacity_cpu = np.asarray(data["excessive_capacity_cpu"], dtype=np.float32)
excessive_capacity_cpu_norm = normalize_to_0_100(excessive_capacity_cpu)

series = ExcessiveCapacitySeries(excessive_capacity_cpu_norm)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    raise ImportError("缺少 matplotlib：请先安装 matplotlib") from e

steps = np.arange(excessive_capacity_cpu.size, dtype=np.int32)
times = steps.astype(np.float64)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(times, excessive_capacity_cpu_norm, linewidth=1.0, color="#d62728")
ax.set_xlabel("time steps")
ax.set_ylabel("excessive_capacity_cpu")
ax.grid(True, alpha=0.25)
fig.tight_layout()

base_dir = Path(__file__).resolve().parent.parent
out_dir = base_dir / "dataset"
if not out_dir.exists():
    raise FileNotFoundError(f"dataset 目录不存在：{out_dir}")

out_png = out_dir / f"excessive_capacity_cpu_curve.png"
fig.savefig(str(out_png), dpi=160)
plt.close(fig)