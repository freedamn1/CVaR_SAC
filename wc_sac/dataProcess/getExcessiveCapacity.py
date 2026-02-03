from __future__ import annotations

"""
从 Alibaba cluster traces 构建"每个时间步的总 CPU 使用量"和"excessive capacity"序列。

数据表：
- machine_meta.csv: 机器元数据，包含 machine_id, TIME_STAMP, cpu_num, mem_size, 状态
- machine_usage.csv: 机器使用率，包含 machine_id, TIME_STAMP, cpu_util_percent (0-100，无效值-1/101需过滤)

计算逻辑：
  excessive capacity C_t = 集群总 CPU 容量 - 该时间步总 CPU 负载

对齐方式：
  - 从 machine_meta.csv 读取每台机器的 cpu_num（CPU核心数）作为容量
  - 从 machine_usage.csv 读取每台机器在时间 t 的 cpu_util_percent（百分比，需过滤无效值）
  - 实际 CPU 使用量 = sum_over_machines(cpu_util_percent[t] / 100 * cpu_num[machine])
  - 集群总容量 = sum_over_available_machines(cpu_num)
  - excessive_capacity_cpu(t) = max(cluster_capacity_cpu - total_usage_cpu(t), 0)
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Literal

import numpy as np


@dataclass(frozen=True)
class ExcessiveCapacitySeries:
    """CPU excessive capacity 序列（按 dt_seconds 对齐）。"""

    dt_seconds: int
    capacity_cpu: float
    # 每步窗口起始时间（秒）
    times: np.ndarray  # shape (T,), int64
    # 每步总 CPU 使用量（CPU 核心数加总；单位=core）
    usage_cpu: np.ndarray  # shape (T,), float32
    # 每步 excessive capacity（CPU 核心数；单位=core）
    excessive_capacity_cpu: np.ndarray  # shape (T,), float32

    def save_npz(self, path: Path) -> None:
        np.savez_compressed(
            str(path),
            dt_seconds=np.int64(self.dt_seconds),
            capacity_cpu=np.float64(self.capacity_cpu),
            times=self.times.astype(np.int64),
            usage_cpu=self.usage_cpu.astype(np.float32),
            excessive_capacity_cpu=self.excessive_capacity_cpu.astype(np.float32),
        )

def _load_machine_capacities(data_dir: Path) -> Dict[str, float]:
    """
    从 machine_meta.csv 加载每台机器的 CPU 容量（cpu_num）。

    你的数据特点：
    - 无表头
    - 列顺序：machine_id, TIME_STAMP, disaster_level_1, disaster_level_2, cpu_num, mem_size, 状态

    规则：
    - 仅保留 状态 == "USING" 的机器
    - 同一机器可能出现多行：取 cpu_num 的最大值（通常 cpu_num 不变）
    """
    csv_path = data_dir / "machine_meta.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 {csv_path}")

    capacities: Dict[str, float] = {}
    with csv_path.open("rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            # machine_id, TIME_STAMP, d1, d2, cpu_num, mem_size, 状态
            if len(row) < 7:
                continue
            machine_id = row[0].strip()
            status = row[6].strip().upper()
            if not machine_id or status != "USING":
                continue
            try:
                cpu_num = int(float(row[4]))
            except Exception:
                continue
            if cpu_num <= 0:
                continue
            prev = capacities.get(machine_id)
            if prev is None or cpu_num > prev:
                capacities[machine_id] = float(cpu_num)

    if not capacities:
        raise RuntimeError("machine_meta.csv 中未找到任何 USING 机器（cpu_num > 0）")

    return capacities


def _align_by_dt(times_arr: np.ndarray, usage_arr: np.ndarray, dt_seconds: int) -> Tuple[np.ndarray, np.ndarray]:
    """把稀疏时间戳序列对齐到等间隔 dt，并对落入同一 bucket 的值求和。"""
    if len(times_arr) < 2:
        raise RuntimeError("时间步数量不足（<2），无法对齐")
    step = int(dt_seconds)
    t0 = int(times_arr.min())
    t_max = int(times_arr.max())
    T = ((t_max - t0) // step) + 1
    usage_aligned = np.zeros((T,), dtype=np.float32)
    times_aligned = (t0 + np.arange(T, dtype=np.int64) * step).astype(np.int64)
    for t_orig, u_val in zip(times_arr, usage_arr):
        idx = (int(t_orig) - t0) // step
        if 0 <= idx < T:
            usage_aligned[int(idx)] += float(u_val)
    return times_aligned, usage_aligned


def _load_usage_series_10sec(
    data_dir: Path,
    machine_capacities: Dict[str, float],
    raw_step_seconds: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 machine_usage.csv 加载“10 秒时间步”的总 CPU 使用量序列（所有机器加总）。

    参数：
    - data_dir: 数据目录
    - machine_capacities: machine_id -> cpu_num 映射
    - raw_step_seconds: 原始时间步长度（秒），默认 10（你当前数据 TIME_STAMP 差分主要为 10）

    返回：(times, usage_cpu)
    - times: 时间戳数组（秒，等间隔 raw_step_seconds）
    - usage_cpu: 每步总 CPU 使用量（CPU核心数，跨机器求和）
    """
    csv_path = data_dir / "machine_usage.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 {csv_path}")

    # 无表头：machine_id, TIME_STAMP, cpu_util_percent, ...
    sums_by_t: Dict[int, float] = {}
    min_t: Optional[int] = None
    max_t: Optional[int] = None
    step = int(raw_step_seconds)
    if step <= 0:
        raise ValueError("raw_step_seconds 必须为正整数。")

    with csv_path.open("rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            machine_id = row[0].strip()
            if not machine_id:
                continue
            cap = machine_capacities.get(machine_id)
            if cap is None:
                continue

            try:
                t = int(float(row[1]))
                cpu_util = int(float(row[2]))
            except Exception:
                continue

            # 过滤无效值（你描述的 -1 或 101，以及任何超出范围）
            if cpu_util < 0 or cpu_util > 100:
                continue

            # 0 表示在 8 天时间跨度之前/之后：对齐时直接跳过
            if t == 0:
                continue

            u = (cpu_util / 100.0) * float(cap)
            sums_by_t[t] = sums_by_t.get(t, 0.0) + u

            if min_t is None or t < min_t:
                min_t = t
            if max_t is None or t > max_t:
                max_t = t

    if min_t is None or max_t is None:
        raise RuntimeError("machine_usage.csv 解析后没有得到任何有效的使用数据")

    # 用连续等间隔的 10 秒时间轴对齐（缺失步补 0），便于后续按 300 秒窗口聚合
    start = int(min_t)
    end = int(max_t)
    if (start % step) != 0:
        # 不强行对齐到 0，只对齐到 step 的倍数，避免引入额外前缀空窗
        start = (start // step) * step
    if (end % step) != 0:
        end = (end // step) * step
    times = np.arange(start, end + step, step, dtype=np.int64)
    usage = np.zeros((len(times),), dtype=np.float32)
    for i, t_val in enumerate(times):
        u = sums_by_t.get(int(t_val), 0.0)
        usage[i] = float(u)

    return times, usage


def _downsample_usage_mean(
    times: np.ndarray,
    usage: np.ndarray,
    raw_step_seconds: int,
    target_dt_seconds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 10 秒使用序列按 target_dt_seconds 做窗口均值下采样。

    - times 必须等间隔 raw_step_seconds
    - 仅保留完整窗口（末尾不足一个窗口的部分丢弃）
    """
    raw_step = int(raw_step_seconds)
    target_dt = int(target_dt_seconds)
    if raw_step <= 0 or target_dt <= 0:
        raise ValueError("raw_step_seconds/target_dt_seconds 必须为正整数。")
    if target_dt % raw_step != 0:
        raise ValueError(f"target_dt_seconds({target_dt}) 必须是 raw_step_seconds({raw_step}) 的整数倍。")

    ratio = target_dt // raw_step
    n = int(len(usage))
    n_win = n // ratio
    if n_win < 1:
        raise RuntimeError("数据长度不足以构建任何一个完整的下采样窗口。")

    cut = n_win * ratio
    usage_cut = usage[:cut].astype(np.float32, copy=False)
    times_cut = times[:cut].astype(np.int64, copy=False)

    usage_ds = usage_cut.reshape(n_win, ratio).mean(axis=1).astype(np.float32, copy=False)
    times_ds = times_cut[::ratio].astype(np.int64, copy=False)
    return times_ds, usage_ds


def build_excessive_capacity_cpu_series(
    data_dir: Path,
    dt_seconds: int = 300,
    capacity_cpu: Optional[float] = None,
    raw_step_seconds: int = 10,
    downsample: Optional[Literal["mean"]] = "mean",
) -> ExcessiveCapacitySeries:
    """
    构建 (times, total_usage_cpu, excessive_capacity_cpu)。

    参数：
    - data_dir: 含 machine_meta.csv 与 machine_usage.csv 的目录（可递归）
    - dt_seconds: 时间步长度（秒），默认 300
    - capacity_cpu: 可选手动指定集群 CPU 总容量（CPU核心数总和）。不指定则从 machine_meta.csv 计算所有可用机器的 cpu_num 总和。
    - raw_step_seconds: machine_usage.csv 的原始时间步（秒），你当前数据为 10
    - downsample: 当 dt_seconds > raw_step_seconds 时的聚合方式；目前仅支持 "mean"
    """
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir 不存在：{data_dir}")

    # 1) 加载机器容量
    machine_capacities = _load_machine_capacities(data_dir)
    total_capacity = sum(machine_capacities.values())

    if capacity_cpu is None:
        capacity_cpu = float(total_capacity)
    else:
        capacity_cpu = float(capacity_cpu)

    # 2) 先加载 10 秒级别使用序列
    times_10, usage_10 = _load_usage_series_10sec(
        data_dir,
        machine_capacities=machine_capacities,
        raw_step_seconds=int(raw_step_seconds),
    )

    # 3) 需要的话再按 dt_seconds 下采样
    if int(dt_seconds) == int(raw_step_seconds):
        times, usage = times_10, usage_10
    else:
        if downsample != "mean":
            raise ValueError("目前仅支持 downsample='mean'。")
        times, usage = _downsample_usage_mean(
            times_10,
            usage_10,
            raw_step_seconds=int(raw_step_seconds),
            target_dt_seconds=int(dt_seconds),
        )

    # 4) 计算 excessive capacity
    excessive_capacity = np.maximum(capacity_cpu - usage.astype(np.float32), 0.0).astype(np.float32)

    return ExcessiveCapacitySeries(
        dt_seconds=int(dt_seconds),
        capacity_cpu=float(capacity_cpu),
        times=times,
        usage_cpu=usage.astype(np.float32),
        excessive_capacity_cpu=excessive_capacity,
    )


def build_excessive_capacity_cpu_series_two_stage(
    data_dir: Path,
    capacity_cpu: Optional[float] = None,
    raw_step_seconds: int = 10,
    env_dt_seconds: int = 300,
) -> Tuple[ExcessiveCapacitySeries, ExcessiveCapacitySeries]:
    """
    两阶段构建：
    1) 先按 raw_step_seconds（默认 10 秒）汇总 usage_cpu，并保存/返回 10 秒级别序列
    2) 再按 env_dt_seconds（默认 300 秒）做窗口均值，返回环境用序列
    """
    series_10 = build_excessive_capacity_cpu_series(
        data_dir=data_dir,
        dt_seconds=int(raw_step_seconds),
        capacity_cpu=capacity_cpu,
        raw_step_seconds=int(raw_step_seconds),
        downsample="mean",
    )
    series_env = build_excessive_capacity_cpu_series(
        data_dir=data_dir,
        dt_seconds=int(env_dt_seconds),
        capacity_cpu=series_10.capacity_cpu,
        raw_step_seconds=int(raw_step_seconds),
        downsample="mean",
    )
    return series_10, series_env
