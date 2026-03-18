from __future__ import annotations

import faiss
import numpy as np

from .common import Timer, ensure_float32_contiguous


def compute_ground_truth_flat(
    xb: np.ndarray,
    xq: np.ndarray,
    k: int,
    metric: str = "ip",
) -> tuple[np.ndarray, np.ndarray, float]:
    xb = ensure_float32_contiguous(xb)
    xq = ensure_float32_contiguous(xq)
    d = xb.shape[1]
    if metric == "ip":
        index = faiss.IndexFlatIP(d)
    elif metric == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    index.add(xb)
    with Timer() as t:
        D, I = index.search(xq, k)
    return D, I, t.seconds


def recall_at_k(i_gt: np.ndarray, i_ann: np.ndarray, k: int) -> float:
    if i_gt.shape != i_ann.shape:
        raise ValueError(f"Shape mismatch: gt={i_gt.shape} ann={i_ann.shape}")
    q = i_gt.shape[0]
    hits = 0.0
    for row in range(q):
        gt_set = set(i_gt[row, :k].tolist())
        ann_set = set(i_ann[row, :k].tolist())
        hits += len(gt_set.intersection(ann_set)) / float(k)
    return hits / float(q)

