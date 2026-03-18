from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil


def ensure_float32_contiguous(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.astype("float32"))


def l2_normalize_inplace(x: np.ndarray) -> np.ndarray:
    import faiss

    x = ensure_float32_contiguous(x)
    faiss.normalize_L2(x)
    return x


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def percentile_ms(samples_s: list[float], p: float) -> float:
    if not samples_s:
        return float("nan")
    return float(np.percentile(np.asarray(samples_s) * 1000.0, p))


def qps(total_queries: int, total_seconds: float) -> float:
    if total_seconds <= 0:
        return 0.0
    return total_queries / total_seconds


def rss_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def write_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


@dataclass
class Timer:
    start: float = 0.0
    end: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end = time.perf_counter()

    @property
    def seconds(self) -> float:
        return self.end - self.start

