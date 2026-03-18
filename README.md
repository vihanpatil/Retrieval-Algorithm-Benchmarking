# Retrieval Algorithm Benchmarking

Lightweight benchmarking harness for comparing dense approximate nearest neighbor (ANN) indexes used in retrieval and RAG-style pipelines.

The project runs parameter sweeps over multiple backends, measures retrieval quality and performance, and exports machine-readable results plus summary plots.

## What It Does

- Loads either synthetic embeddings or NumPy-based datasets
- Computes exact ground truth with a FAISS flat index
- Benchmarks configurable ANN backends across build-time and query-time grids
- Logs recall, latency, QPS, build time, memory usage, and serialized index size to JSONL
- Generates a CSV summary and Pareto-style scatter plots

## Supported Backends

- `faiss_flat`
- `faiss_hnsw`
- `hnswlib`
- `faiss_ivfpq`
- `faiss_opq_ivfpq_factory`

## Repository Layout

```text
bench/
  cli.py          # CLI entrypoints: run and report
  datasets.py     # Synthetic and NumPy dataset loaders
  backends.py     # Index backend implementations
  ground_truth.py # Exact search + recall computation
  runner.py       # Benchmark orchestration and metric collection
  report.py       # CSV and plot generation
configs/
  synthetic.yaml  # Self-contained benchmark config
  rootwise.yaml   # NumPy dataset config for RootWise embeddings
  zonewise.yaml   # NumPy dataset config for ZoneWise embeddings
docs/
  benchmark_roadmap.md
```

## How It Works

1. A YAML config defines the dataset, runtime settings, output paths, and index parameter grids.
2. The runner expands each backend's build and runtime grids into benchmark runs.
3. Exact top-k neighbors are computed once with FAISS flat search and optionally cached.
4. Each ANN configuration is built, queried, and scored against the ground truth.
5. Results are appended as JSONL rows and can later be converted into reports.

## Metrics Captured

- `recall_at_k`
- `latency_ms` (`p50`, `p95`, `p99`)
- `qps`
- build time (`ground_truth`, `train`, `add`)
- resident memory before and after build
- serialized index size

## Configuration

Each config contains:

- `dataset`: synthetic generation or NumPy file paths
- `ground_truth`: optional cache path for exact neighbors
- `runtime`: `k`, warmup queries, and FAISS thread count
- `output`: JSONL destination
- `indexes`: backend definitions with `build_grid` and `runtime_grid`

`configs/synthetic.yaml` is the easiest starting point because it does not depend on external files.

## Getting Started

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a benchmark sweep:

```bash
python -m bench.cli run --config configs/synthetic.yaml
```

Build a report from JSONL results:

```bash
python -m bench.cli report \
  --results artifacts/results/results.jsonl \
  --outdir artifacts/plots
```

## Outputs

- JSONL benchmark records under `artifacts/results/`
- optional cached ground truth under `artifacts/ground_truth/`
- `summary.csv` and plot images under the chosen report directory

## Notes

- `rootwise.yaml` and `zonewise.yaml` currently reference absolute dataset paths outside this repository. Update those paths before running them on another machine.
- `hnswlib` is optional unless you want to benchmark the `hnswlib` backend.
