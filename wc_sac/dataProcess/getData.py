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

from getExcessiveCapacity import build_excessive_capacity_cpu_series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="D:\project\python_project\wxsac\CVaR_SAC\wc_sac\dataset",  help="已下载并解压的数据目录")
    parser.add_argument("--dt_seconds", type=int, default=300, help="时间步长度（秒）")
    parser.add_argument("--out_npz", type=str, default="excessive_capacity_cpu.npz", help="输出文件（npz，含 times/usage/capacity/excessive_capacity）")
    parser.add_argument(
        "--capacity_cpu",
        type=float,
        default=None,
        help="可选：手动指定集群 CPU 总容量（归一化）；不指定则从数据自动计算",
    )
    args = parser.parse_args()

    out_path = Path(args.out_npz).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[info] 使用 Alibaba 2018 数据源")
    result = build_excessive_capacity_cpu_series(
        data_dir=Path(args.data_dir),
        dt_seconds=int(args.dt_seconds),
        capacity_cpu=args.capacity_cpu,
    )
    result.save_npz(out_path)
    print(f"[ok] saved: {out_path}")
    print(f"[info] capacity_cpu={result.capacity_cpu:.6f}, steps={len(result.times)}")


if __name__ == "__main__":
    main()
