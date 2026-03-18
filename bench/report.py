from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _flatten_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for r in rows:
        flat.append(
            {
                "timestamp": r["timestamp"],
                "dataset": r["dataset"],
                "metric": r["metric"],
                "index_type": r["index_type"],
                "build_params": json.dumps(r["build_params"], sort_keys=True),
                "runtime_params": json.dumps(r["runtime_params"], sort_keys=True),
                "k": r["k"],
                "recall_at_k": r["recall_at_k"],
                "p50_ms": r["latency_ms"]["p50"],
                "p95_ms": r["latency_ms"]["p95"],
                "p99_ms": r["latency_ms"]["p99"],
                "qps": r["qps"],
                "train_s": r["build_s"]["train"],
                "add_s": r["build_s"]["add"],
                "rss_after_mb": r["memory_mb"]["rss_after_build"],
                "serialized_mb": r["serialized_bytes"] / (1024 * 1024),
            }
        )
    return pd.DataFrame(flat)


def _scatter(df: pd.DataFrame, x: str, y: str, outpath: Path, title: str) -> None:
    plt.figure(figsize=(10, 6))
    for idx, group in df.groupby("index_type"):
        plt.scatter(group[x], group[y], label=idx, alpha=0.8)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=160)
    plt.close()


def build_report(results_jsonl: str, outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows in {results_jsonl}")
    df = _flatten_rows(rows)
    df.to_csv(out / "summary.csv", index=False)
    _scatter(df, "p95_ms", "recall_at_k", out / "recall_vs_p95.png", "Recall@K vs p95 latency")
    _scatter(df, "qps", "recall_at_k", out / "recall_vs_qps.png", "Recall@K vs QPS")
    _scatter(df, "train_s", "recall_at_k", out / "recall_vs_train_s.png", "Recall@K vs train time")
    _scatter(df, "serialized_mb", "recall_at_k", out / "recall_vs_index_size.png", "Recall@K vs index size")
    print(f"Wrote {out / 'summary.csv'} and plots to {out}")

