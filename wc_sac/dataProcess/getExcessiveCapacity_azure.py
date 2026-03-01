from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

from getExcessiveCapacity import ExcessiveCapacitySeries


COL_CREATED = "vmcreated"
COL_DELETED = "vmdeleted"
COL_AVG_CPU = "avgcpu"
COL_MAX_CPU = "maxcpu"
COL_CORES = "vmcorecount"
REQUIRED_COLUMNS = {COL_CREATED, COL_DELETED, COL_AVG_CPU, COL_MAX_CPU, COL_CORES}
AZURE_HEADERS: List[str] = [
    "vmid",
    "subscriptionid",
    "deploymentid",
    "vmcreated",
    "vmdeleted",
    "maxcpu",
    "avgcpu",
    "p95maxcpu",
    "vmcategory",
    "vmcorecount",
    "vmmemory",
]

def _parse_timestamp(value: str) -> int:
    s = str(value).strip()
    if not s:
        raise ValueError("empty timestamp")
    try:
        v = float(s)
        if v > 1e12 or v > 1e10:
            v = v / 1000.0
        return int(v)
    except Exception:
        pass
    s = s.replace("Z", "+00:00")
    try:
        return int(datetime.fromisoformat(s).timestamp())
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        try:
            return int(datetime.strptime(s, fmt).timestamp())
        except Exception:
            continue
    raise ValueError(f"invalid timestamp: {value}")


def _parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _validate_row_shape(row: dict) -> None:
    if not row:
        raise RuntimeError("vmtable 为空")
    missing = [c for c in REQUIRED_COLUMNS if c not in row]
    if missing:
        raise RuntimeError(f"vmtable 缺少字段: {missing}")


def _open_vmtable_reader(path: Path):
    f = path.open("rt", encoding="utf-8", newline="")
    reader = csv.DictReader(f, fieldnames=AZURE_HEADERS)
    first_row = next(reader, None)
    if first_row is None:
        f.close()
        raise RuntimeError("vmtable 为空")
    reader.fieldnames = AZURE_HEADERS
    return f, reader, first_row


def _iter_vmtable_rows(path: Path):
    f, reader, first_row = _open_vmtable_reader(path)
    try:
        yield first_row
        for row in reader:
            yield row
    finally:
        f.close()


def _scan_time_range(path: Path) -> Tuple[int, int]:
    min_start: Optional[int] = None
    max_end: Optional[int] = None
    for row in _iter_vmtable_rows(path):
        _validate_row_shape(row)
        try:
            start = _parse_timestamp(row.get(COL_CREATED, ""))
            end = _parse_timestamp(row.get(COL_DELETED, ""))
        except Exception:
            continue
        if end <= start:
            continue
        if min_start is None or start < min_start:
            min_start = start
        if max_end is None or end > max_end:
            max_end = end
    if min_start is None or max_end is None:
        raise RuntimeError("无法从数据中解析有效时间范围")
    return int(min_start), int(max_end)


def build_excessive_capacity_cpu_series_azure(
    vmtable_path: Path,
    dt_seconds: int = 300,
) -> ExcessiveCapacitySeries:
    vmtable_path = Path(vmtable_path).expanduser().resolve()
    if not vmtable_path.exists():
        raise FileNotFoundError(f"vmtable 不存在：{vmtable_path}")

    dt = int(dt_seconds)
    if dt <= 0:
        raise ValueError("dt_seconds 必须为正整数")

    min_start, max_end = _scan_time_range(vmtable_path)
    t0 = (min_start // dt) * dt
    total_steps = (max_end - t0) // dt
    if total_steps < 1:
        raise RuntimeError("数据长度不足以形成任何完整的时间步")

    usage_cpu = np.zeros((total_steps,), dtype=np.float32)
    capacity_cpu_series = np.zeros((total_steps,), dtype=np.float32)

    for row in _iter_vmtable_rows(vmtable_path):
        _validate_row_shape(row)
        try:
            start = _parse_timestamp(row.get(COL_CREATED, ""))
            end = _parse_timestamp(row.get(COL_DELETED, ""))
        except Exception:
            continue
        if end <= start:
            continue

        avg_cpu_val = _parse_float(row.get(COL_AVG_CPU, ""))
        max_cpu_val = _parse_float(row.get(COL_MAX_CPU, ""))
        cores_val = _parse_float(row.get(COL_CORES, ""))
        if avg_cpu_val is None or cores_val is None:
            continue
        if max_cpu_val is None:
            max_cpu_val = 100.0

        usage_val = float(avg_cpu_val) / 100.0 * float(cores_val)
        capacity_val = float(max_cpu_val) / 100.0 * float(cores_val)

        start_idx = int(np.ceil((start - t0) / dt))
        end_idx = int(np.floor((end - t0) / dt))
        if end_idx <= start_idx:
            continue
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, total_steps)
        if end_idx <= start_idx:
            continue
        usage_cpu[start_idx:end_idx] += usage_val
        capacity_cpu_series[start_idx:end_idx] += capacity_val

    excessive_capacity_cpu = np.maximum(capacity_cpu_series - usage_cpu, 0.0).astype(np.float32)
    capacity_cpu = float(np.max(capacity_cpu_series)) if capacity_cpu_series.size > 0 else 0.0
    times = (t0 + np.arange(total_steps, dtype=np.int64) * dt).astype(np.int64)

    return ExcessiveCapacitySeries(
        dt_seconds=int(dt),
        capacity_cpu=float(capacity_cpu),
        times=times,
        usage_cpu=usage_cpu.astype(np.float32),
        excessive_capacity_cpu=excessive_capacity_cpu,
    )
