from __future__ import annotations

import csv
import subprocess
from io import StringIO
from typing import Any


def get_gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_installed": False,
        "torch_version": "未安裝",
        "cuda_available": False,
        "torch_cuda_version": "—",
        "gpu_count": 0,
        "gpus": [],
        "error": "",
    }
    try:
        import torch

        info["torch_installed"] = True
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = torch.version.cuda or "—"
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                info["gpus"].append(
                    {
                        "index": index,
                        "name": props.name,
                        "vram_total_mb": round(props.total_memory / 1024**2),
                        "vram_allocated_mb": round(torch.cuda.memory_allocated(index) / 1024**2),
                        "vram_reserved_mb": round(torch.cuda.memory_reserved(index) / 1024**2),
                    }
                )
    except Exception as exc:
        info["error"] = str(exc)

    smi = _query_nvidia_smi()
    for row in smi:
        idx = int(row.get("index", -1))
        target = next((gpu for gpu in info["gpus"] if gpu["index"] == idx), None)
        if target is None:
            target = {"index": idx, "name": row.get("name", "NVIDIA GPU")}
            info["gpus"].append(target)
        target.update(row)
    if info["gpus"]:
        info["gpu_count"] = len(info["gpus"])
    return info


def _query_nvidia_smi() -> list[dict[str, Any]]:
    fields = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=4,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
        if completed.returncode != 0:
            return []
        rows = []
        for values in csv.reader(StringIO(completed.stdout)):
            if len(values) != 7:
                continue
            index, name, util, used, total, temp, power = [item.strip() for item in values]
            rows.append(
                {
                    "index": int(index),
                    "name": name,
                    "utilization_percent": _number(util),
                    "memory_used_mb": _number(used),
                    "memory_total_mb": _number(total),
                    "temperature_c": _number(temp),
                    "power_w": _number(power),
                }
            )
        return rows
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def _number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None

