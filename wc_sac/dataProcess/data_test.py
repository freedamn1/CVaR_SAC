import numpy as np
from pathlib import Path
from wc_sac.envs.preemptive_pricing_env import (
    ExcessiveCapacitySeries,
    PreemptivePricingEnv,
    PricingEnvConfig,
)
# npz keys: ['dt_seconds', 'capacity_cpu', 'times', 'usage_cpu', 'excessive_capacity_cpu']
# total capacityy = 387168.0
# 1) 加载 npz（文件需包含 'excessive_capacity_cpu' 或 'surplus_cpu'）
path = Path(r"D:\project\Python_project\wcsac\WCSAC\wc_sac\dataProcess\excessive_capacity_cpu.npz")

# 详细检查 npz 内容
data = np.load(str(path), allow_pickle=True)
import numpy as _np
# 更高精度打印
_np.set_printoptions(precision=12, suppress=False, linewidth=200)

print("npz keys:", data.files)
usage = np.asarray(data["usage_cpu"], dtype=np.float32) / 100
excessive = np.asarray(data["capacity_cpu"]) - usage
print(excessive.min(), excessive.max(), excessive.mean())

series = ExcessiveCapacitySeries(excessive)

print("excessive_capacity_cpu:", series.excessive_capacity_cpu)
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