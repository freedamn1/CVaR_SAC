from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from .getExcessiveCapacity import ExcessiveCapacitySeries
except Exception:
    from getExcessiveCapacity import ExcessiveCapacitySeries


DEFAULT_SCENARIOS = [
    {"name": "low_load", "mean": 20.0, "var": 5.0, "steps": 96},
    {"name": "mid_load", "mean": 50.0, "var": 15.0, "steps": 96},
    {"name": "high_load", "mean": 80.0, "var": 10.0, "steps": 96},
]


def parse_scenarios(scenarios_json: str | None) -> list[dict]:
    if scenarios_json is None:
        scenarios = list(DEFAULT_SCENARIOS)
    else:
        parsed = json.loads(scenarios_json)
        if not isinstance(parsed, list):
            raise ValueError("scenarios_json 必须是字典列表。")
        scenarios = parsed
    if not scenarios:
        raise ValueError("场景列表不能为空。")
    normalized = []
    for i, sc in enumerate(scenarios):
        if not isinstance(sc, dict):
            raise ValueError(f"第 {i} 个场景不是字典。")
        if "mean" not in sc or "var" not in sc or "steps" not in sc:
            raise ValueError(f"第 {i} 个场景缺少 mean/var/steps。")
        mean = float(sc["mean"])
        var = float(sc["var"])
        steps = int(sc["steps"])
        if var < 0:
            raise ValueError(f"第 {i} 个场景 var 不能为负数。")
        if steps <= 0:
            raise ValueError(f"第 {i} 个场景 steps 必须为正整数。")
        name = str(sc.get("name", f"scenario_{i}"))
        normalized.append({"name": name, "mean": mean, "var": var, "steps": steps})
    return normalized


def simulate_usage(scenarios: list[dict], seed: int, clip_min: float) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    chunks = []
    for sc in scenarios:
        mean = float(sc["mean"])
        std = float(np.sqrt(float(sc["var"])))
        steps = int(sc["steps"])
        chunk = rng.normal(loc=mean, scale=std, size=steps).astype(np.float32)
        chunk = np.maximum(chunk, float(clip_min)).astype(np.float32)
        chunks.append(chunk)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios_json", type=str, default=None, help="场景字典列表 JSON 字符串")
    parser.add_argument("--dt_seconds", type=int, default=300)
    parser.add_argument("--capacity_cpu", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip_min", type=float, default=0.0)
    parser.add_argument("--out_npz", type=str, default=None)
    args = parser.parse_args()

    if int(args.dt_seconds) <= 0:
        raise ValueError("dt_seconds 必须为正整数。")
    if float(args.capacity_cpu) <= 0:
        raise ValueError("capacity_cpu 必须为正数。")

    scenarios = parse_scenarios(args.scenarios_json)
    usage_cpu = simulate_usage(scenarios=scenarios, seed=int(args.seed), clip_min=float(args.clip_min))
    excessive_capacity_cpu = np.maximum(float(args.capacity_cpu) - usage_cpu, 0.0).astype(np.float32)
    times = (np.arange(usage_cpu.size, dtype=np.int64) * int(args.dt_seconds)).astype(np.int64)

    series = ExcessiveCapacitySeries(
        dt_seconds=int(args.dt_seconds),
        capacity_cpu=float(args.capacity_cpu),
        times=times,
        usage_cpu=usage_cpu.astype(np.float32),
        excessive_capacity_cpu=excessive_capacity_cpu.astype(np.float32),
    )

    base_dir = Path(__file__).resolve().parent.parent
    if args.out_npz is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = base_dir / "dataset" / f"simulate_workload_{ts}.npz"
    else:
        out_path = Path(args.out_npz).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    series.save_npz(out_path)

    print("[summary] scenarios")
    for sc in scenarios:
        print(
            f"  {sc['name']}: mean={float(sc['mean']):.6f}, var={float(sc['var']):.6f}, steps={int(sc['steps'])}"
        )
    print(
        f"[info] dt_seconds={int(args.dt_seconds)}, capacity_cpu={float(args.capacity_cpu):.6f}, "
        f"total_steps={int(usage_cpu.size)}"
    )
    print(f"[ok] saved npz: {out_path}")


if __name__ == "__main__":
    main()
