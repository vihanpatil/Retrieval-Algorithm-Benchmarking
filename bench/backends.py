from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .common import Timer, ensure_float32_contiguous

try:
    import hnswlib
except ImportError:  # pragma: no cover
    hnswlib = None


def _faiss_metric(metric: str) -> int:
    if metric == "ip":
        return faiss.METRIC_INNER_PRODUCT
    if metric == "l2":
        return faiss.METRIC_L2
    raise ValueError(f"Unsupported metric: {metric}")


@dataclass
class BuildArtifact:
    backend: str
    index: Any
    build_stats: dict[str, float]


class DenseBackend:
    name: str

    def build(self, xb: np.ndarray, xtrain: np.ndarray, metric: str, build_params: dict[str, Any]) -> BuildArtifact:
        raise NotImplementedError

    def search(self, artifact: BuildArtifact, xq: np.ndarray, k: int, runtime_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def save(self, artifact: BuildArtifact, path: str | Path) -> None:
        raise NotImplementedError

    def serialized_size_bytes(self, artifact: BuildArtifact) -> int:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "index.bin"
            self.save(artifact, p)
            return p.stat().st_size


class FaissFlatBackend(DenseBackend):
    name = "faiss_flat"

    def build(self, xb: np.ndarray, xtrain: np.ndarray, metric: str, build_params: dict[str, Any]) -> BuildArtifact:
        del xtrain, build_params
        xb = ensure_float32_contiguous(xb)
        d = xb.shape[1]
        index = faiss.IndexFlatIP(d) if metric == "ip" else faiss.IndexFlatL2(d)
        with Timer() as t_add:
            index.add(xb)
        return BuildArtifact(backend=self.name, index=index, build_stats={"train_s": 0.0, "add_s": t_add.seconds})

    def search(self, artifact: BuildArtifact, xq: np.ndarray, k: int, runtime_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        del runtime_params
        xq = ensure_float32_contiguous(xq)
        D, I = artifact.index.search(xq, k)
        return I, D

    def save(self, artifact: BuildArtifact, path: str | Path) -> None:
        faiss.write_index(artifact.index, str(path))


class FaissHNSWBackend(DenseBackend):
    name = "faiss_hnsw"

    def build(self, xb: np.ndarray, xtrain: np.ndarray, metric: str, build_params: dict[str, Any]) -> BuildArtifact:
        del xtrain
        xb = ensure_float32_contiguous(xb)
        d = xb.shape[1]
        M = int(build_params["M"])
        efc = int(build_params.get("ef_construction", 200))
        index = faiss.IndexHNSWFlat(d, M, _faiss_metric(metric))
        index.hnsw.efConstruction = efc
        with Timer() as t_add:
            index.add(xb)
        return BuildArtifact(backend=self.name, index=index, build_stats={"train_s": 0.0, "add_s": t_add.seconds})

    def search(self, artifact: BuildArtifact, xq: np.ndarray, k: int, runtime_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        xq = ensure_float32_contiguous(xq)
        artifact.index.hnsw.efSearch = int(runtime_params.get("ef_search", 64))
        D, I = artifact.index.search(xq, k)
        return I, D

    def save(self, artifact: BuildArtifact, path: str | Path) -> None:
        faiss.write_index(artifact.index, str(path))


class HnswlibBackend(DenseBackend):
    name = "hnswlib"

    def build(self, xb: np.ndarray, xtrain: np.ndarray, metric: str, build_params: dict[str, Any]) -> BuildArtifact:
        del xtrain
        if hnswlib is None:
            raise RuntimeError("hnswlib is not installed")
        xb = ensure_float32_contiguous(xb)
        space = {"ip": "ip", "l2": "l2"}.get(metric, metric)
        if space not in {"ip", "l2", "cosine"}:
            raise ValueError(f"Unsupported hnswlib space: {space}")
        idx = hnswlib.Index(space=space, dim=xb.shape[1])
        idx.init_index(
            max_elements=xb.shape[0],
            M=int(build_params["M"]),
            ef_construction=int(build_params.get("ef_construction", 200)),
            random_seed=int(build_params.get("seed", 123)),
        )
        labels = np.arange(xb.shape[0])
        with Timer() as t_add:
            idx.add_items(xb, labels)
        return BuildArtifact(backend=self.name, index=idx, build_stats={"train_s": 0.0, "add_s": t_add.seconds})

    def search(self, artifact: BuildArtifact, xq: np.ndarray, k: int, runtime_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        xq = ensure_float32_contiguous(xq)
        artifact.index.set_ef(int(runtime_params.get("ef", max(k + 1, 64))))
        labels, distances = artifact.index.knn_query(xq, k=k)
        return labels, distances

    def save(self, artifact: BuildArtifact, path: str | Path) -> None:
        artifact.index.save_index(str(path))


class FaissIVFPQBackend(DenseBackend):
    name = "faiss_ivfpq"

    def build(self, xb: np.ndarray, xtrain: np.ndarray, metric: str, build_params: dict[str, Any]) -> BuildArtifact:
        xb = ensure_float32_contiguous(xb)
        xtrain = ensure_float32_contiguous(xtrain)
        d = xb.shape[1]
        nlist = int(build_params["nlist"])
        pq_m = int(build_params["pq_m"])
        pq_nbits = int(build_params.get("pq_nbits", 8))

        n_train = xtrain.shape[0]
        if n_train < nlist:
            raise ValueError(
                f"IVFPQ config invalid: n_train={n_train} < nlist={nlist}. "
                f"Reduce nlist or provide more training vectors."
            )

        if d % pq_m != 0:
            raise ValueError(
                f"IVFPQ config invalid: dim={d} must be divisible by pq_m={pq_m}"
            )

        metric_id = _faiss_metric(metric)
        quantizer = faiss.IndexFlatIP(d) if metric == "ip" else faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, pq_nbits, metric_id)
        with Timer() as t_train:
            index.train(xtrain)
        with Timer() as t_add:
            index.add(xb)
        return BuildArtifact(
            backend=self.name,
            index=index,
            build_stats={"train_s": t_train.seconds, "add_s": t_add.seconds},
        )

    def search(self, artifact: BuildArtifact, xq: np.ndarray, k: int, runtime_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        xq = ensure_float32_contiguous(xq)
        artifact.index.nprobe = int(runtime_params.get("nprobe", 8))
        D, I = artifact.index.search(xq, k)
        return I, D

    def save(self, artifact: BuildArtifact, path: str | Path) -> None:
        faiss.write_index(artifact.index, str(path))


class FaissOPQIVFPQFactoryBackend(DenseBackend):
    name = "faiss_opq_ivfpq_factory"

    def build(self, xb: np.ndarray, xtrain: np.ndarray, metric: str, build_params: dict[str, Any]) -> BuildArtifact:
        xb = ensure_float32_contiguous(xb)
        xtrain = ensure_float32_contiguous(xtrain)
        d = xb.shape[1]
        factory = str(build_params["factory"])
        index = faiss.index_factory(d, factory, _faiss_metric(metric))
        with Timer() as t_train:
            index.train(xtrain)
        with Timer() as t_add:
            index.add(xb)
        return BuildArtifact(
            backend=self.name,
            index=index,
            build_stats={"train_s": t_train.seconds, "add_s": t_add.seconds},
        )

    def search(self, artifact: BuildArtifact, xq: np.ndarray, k: int, runtime_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        xq = ensure_float32_contiguous(xq)
        ivf = faiss.extract_index_ivf(artifact.index)
        if ivf is not None and "nprobe" in runtime_params:
            ivf.nprobe = int(runtime_params["nprobe"])
        D, I = artifact.index.search(xq, k)
        return I, D

    def save(self, artifact: BuildArtifact, path: str | Path) -> None:
        faiss.write_index(artifact.index, str(path))


BACKENDS: dict[str, DenseBackend] = {
    FaissFlatBackend.name: FaissFlatBackend(),
    FaissHNSWBackend.name: FaissHNSWBackend(),
    HnswlibBackend.name: HnswlibBackend(),
    FaissIVFPQBackend.name: FaissIVFPQBackend(),
    FaissOPQIVFPQFactoryBackend.name: FaissOPQIVFPQFactoryBackend(),
}

