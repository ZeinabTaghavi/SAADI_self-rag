from __future__ import annotations

import platform
import resource
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_cpu_memory_mb() -> float:
    # Linux returns kilobytes; macOS returns bytes. The user environment is Linux for execution.
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system().lower() == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024.0


def _run_nvidia_smi() -> Optional[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def get_gpu_memory_snapshot() -> Dict[str, Any]:
    raw = _run_nvidia_smi()
    if not raw:
        return {"gpus": [], "gpu_memory_mb": None, "gpu_peak_memory_mb": None}

    gpus = []
    total_used = 0.0
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        index, name, used_mb, total_mb = parts
        used_value = float(used_mb)
        total_used += used_value
        gpus.append(
            {
                "gpu_index": index,
                "gpu_name": name,
                "gpu_memory_mb": used_value,
                "gpu_total_memory_mb": float(total_mb),
            }
        )
    return {"gpus": gpus, "gpu_memory_mb": total_used, "gpu_peak_memory_mb": total_used}


def capture_resource_usage(phase: str, query_id: Optional[str], doc_id: Optional[str]) -> Dict[str, Any]:
    gpu = get_gpu_memory_snapshot()
    return {
        "captured_at_utc": utc_now_iso(),
        "phase": phase,
        "query_id": query_id,
        "doc_id": doc_id,
        "cpu_memory_mb": get_cpu_memory_mb(),
        "gpu_memory_mb": gpu["gpu_memory_mb"],
        "gpu_peak_memory_mb": gpu["gpu_peak_memory_mb"],
        "gpu_devices": gpu["gpus"],
    }

