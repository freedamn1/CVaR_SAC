from __future__ import annotations

"""
抢占式云服务动态定价环境（Gym API），用于对接 wc_sac/sac/wcsac.py。

状态：
  s_t = [C_t, N_t]
  - C_t：excessive capacity（NCU），由 traces 驱动
  - N_t：当前运行的抢占式实例数量（定义的抽象“实例数”）

动作：
  a_t：归一化动作 in [-1, 1]，环境内部映射到价格 p_t ∈ [p_min, p_max]，
       并将价格离散化到固定档位（默认 0.1~1.0，步长 0.1，且按 p_min/p_max 过滤可用档位）

动态：
  A_t ~ Poisson(f(p_t) * dt)
  L_t ~ Poisson(g(p_t) * dt)
  P_t = max(A_t - L_t - C_t, 0)
  N_{t+1} = max(N_t + A_t - L_t - P_t, 0)

回报：
  r_t = p_t * N_t * dt

成本（约束信号，WC-SAC 读取 info['cost']）：
  cost_t = P_t / max(N_t, eps)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

try:
    import gym
except Exception as e:
    raise ImportError(
        "无法导入依赖 'gym'。请确保使用相同的 Python 解释器安装 gym==0.15.3，\n"
        "示例：C:/Users/86183/anaconda3/python.exe -m pip install 'gym==0.15.3'"
    ) from e

import numpy as np

def normalize_to_0_100(x: np.ndarray) -> np.ndarray:
    """把一维序列做 min-max 归一化到 [0, 100]。"""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    return ((x - x_min) / (x_max - x_min) * 100.0).astype(np.float32)

@dataclass
class PricingEnvConfig:
    p_min: float
    p_max: float
    dt: float  # seconds
    horizon: int  # steps per episode
    n0: float = 10.0
    eps: float = 1e-8
    seed: Optional[int] = None


class ExcessiveCapacitySeries:
    """仅封装 excessive_capacity_cpu 序列供环境按 t 读取。"""

    def __init__(self, excessive_capacity_cpu: np.ndarray):
        self.excessive_capacity_cpu = np.asarray(excessive_capacity_cpu, dtype=np.float32)
        if self.excessive_capacity_cpu.ndim != 1 or len(self.excessive_capacity_cpu) < 2:
            raise ValueError("excessive_capacity_cpu 必须是一维数组，且长度至少为 2。")

    @classmethod
    def from_npz(cls, path: str | Path) -> "ExcessiveCapacitySeries":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"NPZ file not found: {path}")
        data = np.load(str(path), allow_pickle=True)
        # 明确检查键，避免对 numpy 数组做布尔判断
        if "excessive_capacity_cpu" in data:
            excessive_capacity = data["excessive_capacity_cpu"]
        else:
            raise KeyError("NPZ file must contain 'excessive_capacity_cpu' key.")
        return cls(excessive_capacity)

    def __len__(self) -> int:
        return int(len(self.excessive_capacity_cpu))

    def at(self, t: int) -> float:
        t = int(np.clip(t, 0, len(self.excessive_capacity_cpu) - 1))
        return float(self.excessive_capacity_cpu[t])


class PreemptivePricingEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        capacity_series: ExcessiveCapacitySeries,
        cfg: PricingEnvConfig,
        f_arrival_rate: Callable[[float], float],
        g_departure_rate: Callable[[float], float],
    ):
        super().__init__()
        self.series = capacity_series
        self.cfg = cfg
        self.f = f_arrival_rate
        self.g = g_departure_rate

        # 动作：归一化 a_norm ∈ [-1, 1]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.series.excessive_capacity_cpu = normalize_to_0_100(self.series.excessive_capacity_cpu)
        # 观测：C_t, N_t
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        self._rng = np.random.RandomState(cfg.seed)
        self._t = 0
        self._t0 = 0
        self._n = float(cfg.n0)
        self._use_horizon_done = True
        # 离散价格档位：默认 0.1, 0.2, ..., 1.0；并按 p_min/p_max 过滤可用档位
        base_levels = np.round(np.arange(0.1, 1.01, 0.1), 2).astype(np.float32)
        p_min = float(self.cfg.p_min)
        p_max = float(self.cfg.p_max)
        levels = base_levels[(base_levels >= (p_min - 1e-6)) & (base_levels <= (p_max + 1e-6))]
        if levels.size == 0:
            raise ValueError(
                f"离散价格档位 0.1~1.0 与 p_min/p_max 不相交：p_min={p_min}, p_max={p_max}。"
                "请调整 p_min/p_max 使其覆盖至少一个离散价格点（如 0.1）。"
            )
        self._price_levels = levels

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            return
        self._rng = np.random.RandomState(int(seed))

    def _discretize_price(self, price: float) -> float:
        """把连续价格离散到最近的档位（self._price_levels）。"""
        p = float(price)
        idx = int(np.argmin(np.abs(self._price_levels - p)))
        return float(self._price_levels[idx])

    def _action_to_price(self, a_norm: float) -> float:
        """归一化动作 [-1,1] → 连续价格 [p_min,p_max] → 离散化到固定档位。"""
        a = float(np.clip(a_norm, -1.0, 1.0))
        p_cont = float(self.cfg.p_min + (a + 1.0) * 0.5 * (self.cfg.p_max - self.cfg.p_min))
        p_disc = self._discretize_price(p_cont)
        # 最后再保险裁剪到 [p_min, p_max]
        return float(np.clip(p_disc, float(self.cfg.p_min), float(self.cfg.p_max)))

    def reset(self):
        series_len = len(self.series)
        horizon = int(self.cfg.horizon)
        if horizon <= 0:
            raise ValueError(f"horizon 必须为正整数，当前为 {horizon}.")

        max_start = max(series_len - 1 - horizon, 0)
        start_t = int(self._rng.randint(0, max_start + 1)) if max_start > 0 else 0

        self._t0 = start_t
        self._t = start_t
        self._n = float(self.cfg.n0)
        self._use_horizon_done = True
        c0 = self.series.at(self._t)
        return np.array([c0, self._n], dtype=np.float32)

    def reset_for_test(self):
        self._t0 = 0
        self._t = 0
        self._n = float(self.cfg.n0)
        self._use_horizon_done = False
        c0 = self.series.at(self._t)
        return np.array([c0, self._n], dtype=np.float32)

    def step(self, action):
        a_norm = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        price = self._action_to_price(a_norm)

        c_t = self.series.at(self._t)
        n_t = self._n

        lam_a = max(float(self.f(price)) * float(self.cfg.dt), 0.0)
        lam_l = max(float(self.g(price)) * float(self.cfg.dt), 0.0)
        arrivals = float(self._rng.poisson(lam_a))
        leaves = float(self._rng.poisson(lam_l))
        # arrivals = lam_a
        # leaves = lam_l
        preempted = max(n_t + arrivals - leaves - c_t, 0.0)
        n_next = max(n_t + arrivals - leaves - preempted, 0.0)
        reward = price * n_t * float(self.cfg.dt)
        cost = preempted / max(n_next + preempted, float(self.cfg.eps))

        # advance time
        self._t += 1
        self._n = n_next

        done_by_horizon = False
        if self._use_horizon_done:
            done_by_horizon = bool((self._t - self._t0) >= int(self.cfg.horizon))
        done = bool(done_by_horizon or self._t >= (len(self.series) - 1))
        c_next = self.series.at(self._t)
        obs = np.array([c_next, self._n], dtype=np.float32)

        info = {
            "cost": float(cost),
            "price": float(price),
            "arrivals": float(arrivals),
            "leaves": float(leaves),
            "preempted": float(preempted),
            "excessive_capacity": float(c_t),
            "n_running": float(n_t),
        }
        return obs, float(reward), done, info

