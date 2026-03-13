from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gym
import numpy as np


def normalize_to_0_100(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    x_min = float(np.nanmin(arr))
    x_max = float(np.nanmax(arr))
    if abs(x_max - x_min) < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - x_min) / (x_max - x_min) * 100.0).astype(np.float32)


@dataclass
class PricingEnvConfig:
    p_min: float
    p_max: float
    dt: float
    horizon: int
    n0: float = 10.0
    eta: float = 0.33
    thet: float = 0.33
    k: float = 2.0
    omega: float = 2.4
    eps: float = 1e-8
    seed: Optional[int] = None
    price_level_min: float = 0.1
    price_level_max: float = 1.0
    price_level_step: float = 0.1


class ExcessiveCapacitySeries:
    def __init__(self, excessive_capacity_cpu: np.ndarray):
        self.excessive_capacity_cpu = np.asarray(excessive_capacity_cpu, dtype=np.float32)
        if self.excessive_capacity_cpu.ndim != 1 or len(self.excessive_capacity_cpu) < 2:
            raise ValueError("excessive_capacity_cpu 必须是一维数组，且长度至少为 2。")
        self.excessive_capacity_cpu = normalize_to_0_100(self.excessive_capacity_cpu)

    @classmethod
    def from_npz(cls, path: str | Path) -> "ExcessiveCapacitySeries":
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {file_path}")
        data = np.load(str(file_path), allow_pickle=True)
        if "excessive_capacity_cpu" not in data:
            raise KeyError("NPZ file must contain 'excessive_capacity_cpu' key.")
        return cls(data["excessive_capacity_cpu"])

    def __len__(self) -> int:
        return int(len(self.excessive_capacity_cpu))

    def at(self, t: int) -> float:
        idx = int(np.clip(t, 0, len(self.excessive_capacity_cpu) - 1))
        return float(self.excessive_capacity_cpu[idx])


class PreemptivePricingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 1}

    def __init__(self, capacity_series: ExcessiveCapacitySeries, cfg: PricingEnvConfig):
        super().__init__()
        self.series = capacity_series
        self.cfg = cfg
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self._rng = np.random.RandomState(cfg.seed)
        self._t = 0
        self._t0 = 0
        self._n = float(cfg.n0)
        self._use_horizon_done = True
        base_levels = np.round(
            np.arange(cfg.price_level_min, cfg.price_level_max + 1e-8, cfg.price_level_step),
            4,
        ).astype(np.float32)
        levels = base_levels[(base_levels >= (cfg.p_min - 1e-6)) & (base_levels <= (cfg.p_max + 1e-6))]
        if levels.size == 0:
            raise ValueError(
                f"离散价格档位与 p_min/p_max 不相交：p_min={cfg.p_min}, p_max={cfg.p_max}。"
            )
        self._price_levels = levels

    def _f(self, p: float) -> float:
        try:
            val = self.cfg.eta * (1.0 - float(p) ** self.cfg.k) ** self.cfg.omega
        except Exception:
            val = 0.0
        return max(float(val), 0.0)

    def _g(self, p: float) -> float:
        try:
            val = self.cfg.thet - self.cfg.thet * (1.0 - float(p) ** self.cfg.k) ** self.cfg.omega
        except Exception:
            val = 0.0
        return max(float(val), 0.0)

    def _discretize_price(self, price: float) -> float:
        p = float(price)
        idx = int(np.argmin(np.abs(self._price_levels - p)))
        return float(self._price_levels[idx])

    def _action_to_price(self, a_norm: float) -> float:
        a = float(np.clip(a_norm, -1.0, 1.0))
        p_cont = float(self.cfg.p_min + (a + 1.0) * 0.5 * (self.cfg.p_max - self.cfg.p_min))
        p_disc = self._discretize_price(p_cont)
        return float(np.clip(p_disc, float(self.cfg.p_min), float(self.cfg.p_max)))

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(int(seed))
        horizon = int(self.cfg.horizon)
        if horizon <= 0:
            raise ValueError(f"horizon 必须为正整数，当前为 {horizon}.")
        max_start = max(len(self.series) - 1 - horizon, 0)
        start_t = int(self._rng.randint(0, max_start + 1)) if max_start > 0 else 0
        self._t0 = start_t
        self._t = start_t
        self._n = float(self.cfg.n0)
        self._use_horizon_done = True
        c0 = self.series.at(self._t)
        obs = np.array([c0, self._n], dtype=np.float32)
        return obs, {}

    def reset_for_test(self):
        self._t0 = 0
        self._t = 0
        self._n = float(self.cfg.n0)
        self._use_horizon_done = False
        c0 = self.series.at(self._t)
        return np.array([c0, self._n], dtype=np.float32), {}

    def step(self, action):
        a_norm = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        price = self._action_to_price(a_norm)
        c_t = self.series.at(self._t)
        n_t = self._n
        lam_a = max(self._f(price) * float(self.cfg.dt), 0.0)
        lam_l = max(self._g(price) * float(self.cfg.dt), 0.0)
        arrivals = float(self._rng.poisson(lam_a))
        leaves = float(self._rng.poisson(lam_l))
        preempted = max(n_t + arrivals - leaves - c_t, 0.0)
        n_next = max(n_t + arrivals - leaves - preempted, 0.0)
        reward = price * n_t * float(self.cfg.dt)
        cost = preempted / max(n_next + preempted, float(self.cfg.eps))
        self._t += 1
        self._n = n_next
        done_by_horizon = False
        if self._use_horizon_done:
            done_by_horizon = bool((self._t - self._t0) >= int(self.cfg.horizon))
        terminated = bool(done_by_horizon or self._t >= (len(self.series) - 1))
        truncated = False
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
        return obs, float(reward), terminated, truncated, info
