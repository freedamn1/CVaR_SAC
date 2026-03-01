"""
通用数据预处理入口：生成环境需要的“每个时间步 excessive capacity”序列 C_t（保存为 npz）。

当前支持的数据集：
1) alibaba：Alibaba 2017 cluster traces
   - 输入：machine_meta.csv + machine_usage.csv
   - 处理：按 dt_seconds 聚合，直接生成环境用序列（不再输出 10 秒级文件）
2) azure：AzurePublicDatasetV1 (vmtable)
   - 输入：vmtable.csv（含创建/删除时间、avg/max cpu、core 数）
   - 处理：按 300 秒切分，仅累加完整切分区间，avg/max cpu 视为 0-100 百分比

用法：
  python getData.py --dataset alibaba --data_dir <dir> --dt_seconds 300
  python getData.py --dataset azure --vmtable <path> --dt_seconds 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

from getExcessiveCapacity import build_excessive_capacity_cpu_series
from getExcessiveCapacity_azure import build_excessive_capacity_cpu_series_azure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="alibaba", choices=["alibaba", "azure"])
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"D:\project\python_project\wxsac\CVaR_SAC\wc_sac\dataset",
        help="Alibaba 数据目录（含 machine_meta.csv / machine_usage.csv）",
    )
    parser.add_argument(
        "--vmtable",
        type=str,
        default=None,
        help="Azure vmtable CSV 路径",
    )
    parser.add_argument("--raw_step_seconds", type=int, default=10, help="Alibaba 原始时间步（秒）")
    parser.add_argument("--dt_seconds", type=int, default=300, help="环境使用的时间步长度（秒），例如 300")
    parser.add_argument(
        "--out_npz",
        type=str,
        default=None,
        help="环境使用输出文件（npz，含 dt_seconds/capacity_cpu/times/usage_cpu/excessive_capacity_cpu）。默认保存到输入同级目录",
    )
    parser.add_argument(
        "--capacity_cpu",
        type=float,
        default=None,
        help="Alibaba 可选：手动指定集群 CPU 总容量（CPU 核心数总和）；不指定则从数据自动计算",
    )
    args = parser.parse_args()

    if args.dataset == "azure":
        vmtable_path = Path(args.vmtable).expanduser().resolve() if args.vmtable else None
        if vmtable_path is None or not vmtable_path.exists():
            raise FileNotFoundError("Azure 数据集需要提供 --vmtable 路径")
        dataset_dir = vmtable_path.parent
        if args.out_npz is None:
            out_path_env = dataset_dir / "AzurePublicDatasetV1_excessive_capacity_cpu_300sec.npz"
        else:
            out_path_env = Path(args.out_npz).expanduser().resolve()
        out_path_env.parent.mkdir(parents=True, exist_ok=True)

        print("[info] 使用 AzurePublicDatasetV1 数据源")
        series_env = build_excessive_capacity_cpu_series_azure(
            vmtable_path=vmtable_path,
            dt_seconds=int(args.dt_seconds),
        )
        series_env.save_npz(out_path_env)
        print(f"[ok] saved env:   {out_path_env}")
        print(f"[info] env  dt_seconds={series_env.dt_seconds}, capacity_cpu={series_env.capacity_cpu:.6f}, steps={len(series_env.times)}")
    else:
        data_dir = Path(args.data_dir).expanduser().resolve()
        dataset_dir = data_dir
        if args.out_npz is None:
            out_path_env = dataset_dir / "excessive_capacity_cpu_300sec.npz"
        else:
            out_path_env = Path(args.out_npz).expanduser().resolve()
        out_path_env.parent.mkdir(parents=True, exist_ok=True)

        print("[info] 使用 Alibaba 2017 cluster traces 数据源")
        series_env = build_excessive_capacity_cpu_series(
            data_dir=Path(args.data_dir),
            capacity_cpu=args.capacity_cpu,
            raw_step_seconds=int(args.raw_step_seconds),
            dt_seconds=int(args.dt_seconds),
        )
        series_env.save_npz(out_path_env)
        print(f"[ok] saved env:   {out_path_env}")
        print(f"[info] env  dt_seconds={series_env.dt_seconds}, capacity_cpu={series_env.capacity_cpu:.6f}, steps={len(series_env.times)}")


if __name__ == "__main__":
    main()
