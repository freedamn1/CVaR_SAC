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
path = base_dir / "dataset" / "excessive_capacity_cpu_10sec.npz"

# 详细检查 npz 内容
data = np.load(str(path), allow_pickle=True)

excessive_capacity_cpu = np.asarray(data["excessive_capacity_cpu"], dtype=np.float32)
excessive_capacity_cpu_norm = normalize_to_0_100(excessive_capacity_cpu)

series = ExcessiveCapacitySeries(excessive_capacity_cpu_norm)

print(
    "excessive_capacity_cpu normalized[0,100]:",
    f"min={float(series.excessive_capacity_cpu.min()):.6f}, "
    f"max={float(series.excessive_capacity_cpu.max()):.6f}, "
    f"mean={float(series.excessive_capacity_cpu.mean()):.6f}",
)
# # 2) 配置环境参数（示例）
# cfg = PricingEnvConfig(
#     p_min=0.1,
#     p_max=1.0,
#     dt=300.0,      # 与数据的时间步一致（秒）
#     horizon=200,   # 每个 episode 的步数
#     n0=0.0,
#     seed=42,
# )

# # 3) 定义到达与离开速率函数 f(p), g(p)
# #    这里给出简单示例，实际请用你需要的函数
# def f_arrival(p: float) -> float:
#     return max(0.1 * (1.0 - p), 0.0)

# def g_leave(p: float) -> float:
#     return max(0.05 * p, 0.0)

# # 4) 创建环境
# env = PreemptivePricingEnv(capacity_series=series, cfg=cfg, f_arrival_rate=f_arrival, g_leave_rate=g_leave)

# # 5) 快速交互测试
# obs = env.reset()
# done = False
# total_reward = 0.0
# while not done:
#     action = 0.0  # 例如恒定动作（归一化），或随机：env.action_space.sample()
#     obs, reward, done, info = env.step([action])
#     total_reward += reward

# print("episode reward:", total_reward)