from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np

from .backends import BACKENDS
from .common import now_utc_iso, percentile_ms, qps, rss_mb, write_jsonl
from .datasets import maybe_load_ground_truth, maybe_save_ground_truth
from .ground_truth import compute_ground_truth_flat, recall_at_k


def _expand_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    values: list[list[Any]] = []
    for k in keys:
        v = grid[k]
        values.append(v if isinstance(v, list) else [v])
    combos: list[dict[str, Any]] = []
    for tup in itertools.product(*values):
        combos.append({k: tup[i] for i, k in enumerate(keys)})
    return combos


def _iter_runs(cfg: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for index_cfg in cfg["indexes"]:
        build_grid = _expand_grid(index_cfg.get("build_grid", {}))
        runtime_grid = _expand_grid(index_cfg.get("runtime_grid", {}))
        if not build_grid:
            build_grid = [{}]
        if not runtime_grid:
            runtime_grid = [{}]
        for b in build_grid:
            for r in runtime_grid:
                yield {"name": index_cfg["name"], "build_params": b, "runtime_params": r}


def _search_latency_distribution(
    search_fn,
    xq: np.ndarray,
    k: int,
    warmup: int,
) -> tuple[np.ndarray, list[float], float]:
    nq = xq.shape[0]
    warmup_n = min(warmup, nq)
    if warmup_n > 0:
        _ = search_fn(xq[:warmup_n], k)
    latencies: list[float] = []
    all_ids = []
    t_total0 = time.perf_counter()
    for i in range(nq):
        t0 = time.perf_counter()
        ids, _scores = search_fn(xq[i : i + 1], k)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        all_ids.append(ids[0])
    t_total1 = time.perf_counter()
    ann_ids = np.asarray(all_ids, dtype=np.int64)
    return ann_ids, latencies, (t_total1 - t_total0)


def run_benchmark(cfg: dict[str, Any], bundle, out_jsonl: str) -> None:
    runtime = cfg.get("runtime", {})
    k = int(runtime.get("k", 10))
    warmup = int(runtime.get("warmup_queries", 200))
    faiss_threads = runtime.get("faiss_threads")
    if faiss_threads is not None:
        faiss.omp_set_num_threads(int(faiss_threads))
        os.environ["OMP_NUM_THREADS"] = str(faiss_threads)

    gt_cfg = cfg.get("ground_truth", {})
    gt_path = gt_cfg.get("cache_path")
    if gt_path:
        cached = maybe_load_ground_truth(gt_path)
    else:
        cached = None

    if cached is None:
        d_gt, i_gt, gt_s = compute_ground_truth_flat(bundle.xb, bundle.xq, k=k, metric=bundle.metric)
        if gt_path:
            maybe_save_ground_truth(gt_path, d_gt, i_gt)
    else:
        d_gt, i_gt = cached
        gt_s = float("nan")

    git_sha = os.environ.get("GIT_SHA", "unknown")
    seed = int(cfg.get("seed", 123))

    for run in _iter_runs(cfg):
        backend_name = run["name"]
        backend = BACKENDS[backend_name]
        build_params = run["build_params"]
        runtime_params = run["runtime_params"]

        rss_before = rss_mb()
        artifact = backend.build(bundle.xb, bundle.xtrain, bundle.metric, build_params)
        rss_after = rss_mb()
        serialized_bytes = backend.serialized_size_bytes(artifact)

        def search_fn(xqq: np.ndarray, kk: int):
            return backend.search(artifact, xqq, kk, runtime_params)

        ann_ids, latencies_s, total_s = _search_latency_distribution(search_fn, bundle.xq, k, warmup)
        rec = recall_at_k(i_gt, ann_ids, k)

        row = {
            "timestamp": now_utc_iso(),
            "dataset": bundle.name,
            "metric": "cosine_via_ip_norm" if (bundle.metric == "ip" and bundle.normalize) else bundle.metric,
            "index_type": backend_name,
            "build_params": build_params,
            "runtime_params": runtime_params,
            "k": k,
            "recall_at_k": rec,
            "latency_ms": {
                "p50": percentile_ms(latencies_s, 50),
                "p95": percentile_ms(latencies_s, 95),
                "p99": percentile_ms(latencies_s, 99),
            },
            "qps": qps(bundle.xq.shape[0], total_s),
            "build_s": {
                "ground_truth": gt_s,
                "train": artifact.build_stats.get("train_s", 0.0),
                "add": artifact.build_stats.get("add_s", 0.0),
            },
            "memory_mb": {
                "rss_before_build": rss_before,
                "rss_after_build": rss_after,
            },
            "serialized_bytes": serialized_bytes,
            "git_sha": git_sha,
            "seed": seed,
        }
        write_jsonl(out_jsonl, row)
        print(json.dumps(row))

