"""
Alibaba 2017 cluster traces：数据预处理入口。

目标：生成环境需要的“每个时间步 excessive capacity”序列 C_t（保存为 npz）。

输入（自动探测，优先级从高到低）：
1) machine_usage.*（你已准备好的 machine_usage 数据）
   - 支持：machine_usage.csv / machine_usage.csv.gz / machine_usage.npz
2) 原始表：
   - server_usage.csv（机器使用率）
   - server_event.csv（机器容量）

用法：
  python getData.py --data_dir <dir> --out_npz excessive_capacity_cpu.npz --dt_seconds 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

from getExcessiveCapacity import build_excessive_capacity_cpu_series_two_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"D:\project\python_project\wxsac\CVaR_SAC\wc_sac\dataset",
        help="已下载并解压的数据目录（含 machine_meta.csv / machine_usage.csv）",
    )
    parser.add_argument("--raw_step_seconds", type=int, default=10, help="原始时间步（秒），你的 TIME_STAMP 目前是 10 秒")
    parser.add_argument("--dt_seconds", type=int, default=300, help="环境使用的时间步长度（秒），例如 300")
    parser.add_argument(
        "--out_npz_10sec",
        type=str,
        default="excessive_capacity_cpu_10sec.npz",
        help="10 秒级别输出文件（npz，含 dt_seconds/capacity_cpu/times/usage_cpu/excessive_capacity_cpu）",
    )
    parser.add_argument(
        "--out_npz",
        type=str,
        default="excessive_capacity_cpu.npz",
        help="环境使用输出文件（npz，含 dt_seconds/capacity_cpu/times/usage_cpu/excessive_capacity_cpu）",
    )
    parser.add_argument(
        "--capacity_cpu",
        type=float,
        default=None,
        help="可选：手动指定集群 CPU 总容量（CPU 核心数总和）；不指定则从数据自动计算",
    )
    args = parser.parse_args()

    out_path_10 = Path(args.out_npz_10sec).expanduser().resolve()
    out_path_10.parent.mkdir(parents=True, exist_ok=True)
    out_path_env = Path(args.out_npz).expanduser().resolve()
    out_path_env.parent.mkdir(parents=True, exist_ok=True)

    print("[info] 使用 Alibaba 2018 数据源")
    series_10, series_env = build_excessive_capacity_cpu_series_two_stage(
        data_dir=Path(args.data_dir),
        capacity_cpu=args.capacity_cpu,
        raw_step_seconds=int(args.raw_step_seconds),
        env_dt_seconds=int(args.dt_seconds),
    )

    series_10.save_npz(out_path_10)
    print(f"[ok] saved 10sec: {out_path_10}")
    print(f"[info] 10sec dt_seconds={series_10.dt_seconds}, capacity_cpu={series_10.capacity_cpu:.6f}, steps={len(series_10.times)}")

    series_env.save_npz(out_path_env)
    print(f"[ok] saved env:   {out_path_env}")
    print(f"[info] env  dt_seconds={series_env.dt_seconds}, capacity_cpu={series_env.capacity_cpu:.6f}, steps={len(series_env.times)}")


if __name__ == "__main__":
    main()
