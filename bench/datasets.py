from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .common import ensure_float32_contiguous, l2_normalize_inplace


@dataclass
class DatasetBundle:
    name: str
    xb: np.ndarray
    xq: np.ndarray
    xtrain: np.ndarray
    metric: Literal["ip", "l2"]
    normalize: bool


def _maybe_normalize(x: np.ndarray, normalize: bool) -> np.ndarray:
    x = ensure_float32_contiguous(x)
    if normalize:
        x = l2_normalize_inplace(x)
    return x


def load_numpy_bundle(
    name: str,
    xb_path: str,
    xq_path: str,
    xtrain_path: str | None = None,
    metric: Literal["ip", "l2"] = "ip",
    normalize: bool = False,
) -> DatasetBundle:
    xb = np.load(xb_path)
    xq = np.load(xq_path)
    xtrain = np.load(xtrain_path) if xtrain_path else xb
    xb = _maybe_normalize(xb, normalize)
    xq = _maybe_normalize(xq, normalize)
    xtrain = _maybe_normalize(xtrain, normalize)
    return DatasetBundle(name=name, xb=xb, xq=xq, xtrain=xtrain, metric=metric, normalize=normalize)


def make_synthetic_bundle(
    name: str,
    n_base: int,
    n_query: int,
    dim: int,
    n_train: int | None = None,
    seed: int = 123,
    metric: Literal["ip", "l2"] = "ip",
    normalize: bool = True,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    xb = rng.standard_normal((n_base, dim), dtype=np.float32)
    xq = rng.standard_normal((n_query, dim), dtype=np.float32)
    n_train = n_train or min(n_base, 100_000)
    xtrain = xb[:n_train].copy()
    xb = _maybe_normalize(xb, normalize)
    xq = _maybe_normalize(xq, normalize)
    xtrain = _maybe_normalize(xtrain, normalize)
    return DatasetBundle(name=name, xb=xb, xq=xq, xtrain=xtrain, metric=metric, normalize=normalize)


def load_bundle_from_config(cfg: dict) -> DatasetBundle:
    ds = cfg["dataset"]
    kind = ds.get("kind", "synthetic")
    if kind == "synthetic":
        return make_synthetic_bundle(
            name=ds.get("name", "synthetic"),
            n_base=int(ds["n_base"]),
            n_query=int(ds["n_query"]),
            dim=int(ds["dim"]),
            n_train=ds.get("n_train"),
            seed=int(ds.get("seed", 123)),
            metric=ds.get("metric", "ip"),
            normalize=bool(ds.get("normalize", True)),
        )
    if kind == "numpy":
        return load_numpy_bundle(
            name=ds["name"],
            xb_path=ds["xb_path"],
            xq_path=ds["xq_path"],
            xtrain_path=ds.get("xtrain_path"),
            metric=ds.get("metric", "ip"),
            normalize=bool(ds.get("normalize", False)),
        )
    raise ValueError(f"Unsupported dataset kind: {kind}")


def maybe_save_ground_truth(path: str | Path, d: np.ndarray, i: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, D=d, I=i)


def maybe_load_ground_truth(path: str | Path) -> tuple[np.ndarray, np.ndarray] | None:
    path = Path(path)
    if not path.exists():
        return None
    data = np.load(path)
    return data["D"], data["I"]

