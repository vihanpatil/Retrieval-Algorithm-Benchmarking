# Implementation Roadmap

## Phase 0
- Freeze hybrid sparse/fusion logic.
- Build exact baseline and ground truth with `faiss_flat`.
- Enable JSONL logging for recall, latency, QPS, build time, memory, and index size.

## Phase 1
- Run HNSW sweeps (`faiss_hnsw`, `hnswlib`):
  - `M in {8,16,32,48,64}`
  - `ef_construction in {64,128,256,400}`
  - `ef_search / ef in {16,32,64,128,256,512}`

## Phase 2
- Run IVFPQ and OPQ+IVFPQ sweeps:
  - `nlist in {4*sqrt(N), 8*sqrt(N), 16*sqrt(N)}`
  - `nprobe in {1,2,4,8,16,32,64,128}`
  - `pq_m in {8,16,32,64}`
  - `pq_nbits in {4,6,8}`
  - add `RFlat` variants for high-recall rerank.

## Phase 3
- Run final benchmarks on:
  - standard ANN datasets
  - real RAG embeddings
- Generate Pareto plots and pick operating points.

## Commands

```bash
python -m bench.cli run --config configs/synthetic.yaml
python -m bench.cli run --config configs/rag_numpy_template.yaml
python -m bench.cli report --results artifacts/results/results.jsonl --outdir artifacts/plots
```

