"""Spread chunks over several workers with Dask: one worker per GPU.

Why Dask?  A Kaggle "T4 x2" machine has two GPUs.  One Python process can
drive one GPU comfortably; Dask starts one *worker process* per GPU and hands
each of them chunks to search.  The main process only reads the FASTA, sends
chunks out, and collects the (small) hit lists that come back.

    main process                      worker 0 (GPU 0)    worker 1 (GPU 1)
    iter_chunks() ── chunk 0 ──────▶  search
                  ── chunk 1 ─────────────────────────▶  search
                  ◀── hits 0 ───────
                  ── chunk 2 ──────▶  search            (next chunk goes to
                  ◀── hits 1 ───────────────────────────  whoever is free)

* On GPU machines ``LocalCUDACluster()`` (from dask-cuda) makes one worker per
  GPU and pins each worker to its own GPU.
* Everywhere else (e.g. the Mac) a normal CPU ``LocalCluster`` runs the same
  code with the CPU backends, so the scheduling logic can be tested locally.

Memory stays flat: instead of submitting every chunk at once (hg38 at 50 M
bases per chunk would put ~3 GB in flight), at most ``max_in_flight`` chunks
are out at any time.  A new one is sent each time a result comes back.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .cpu_baseline import get_backend
from .encoding import validate_guide
from .parser import DEFAULT_CHUNK_SIZE, iter_chunks
from .results import Hit, dedupe_hits, make_hit, sort_hits


@dataclass
class ChunkResult:
    """What one worker reports back for one chunk."""

    index: int
    chrom: str
    offset: int
    bases: int
    hits: list[Hit]
    seconds: float  # time spent searching on the worker
    worker: str  # Dask worker name
    device: str  # "GPU 1" or "CPU"


def _device_label(backend: str) -> str:
    if backend != "gpu":
        return "CPU"
    # dask-cuda gives each worker CUDA_VISIBLE_DEVICES="<its gpu>,<others...>",
    # so the first entry is the physical GPU this worker uses.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return f"GPU {visible.split(',')[0]}" if visible else "GPU 0"


def search_task(
    index: int, chrom: str, offset: int, seq: np.ndarray, guide: np.ndarray, max_mm: int, backend: str
) -> ChunkResult:
    """Runs on a worker: search one chunk and build the hit records there,
    so only the small hit list travels back, not the chunk."""
    from distributed import get_worker

    t0 = time.perf_counter()
    search = get_backend(backend)
    hits = [make_hit(chrom, offset, seq, pos, strand, mm, guide) for pos, strand, mm in search(seq, guide, max_mm)]
    seconds = time.perf_counter() - t0
    return ChunkResult(index, chrom, offset, len(seq), hits, seconds, get_worker().name, _device_label(backend))


def make_cluster(backend: str, n_workers: int):
    """One worker per GPU for the GPU backend on a real NVIDIA machine;
    otherwise a CPU cluster with one single-threaded process per worker."""
    if backend == "gpu":
        from numba import config, cuda

        if not config.ENABLE_CUDASIM and cuda.is_available():
            from dask_cuda import LocalCUDACluster

            visible = ",".join(str(i) for i in range(n_workers))
            return LocalCUDACluster(CUDA_VISIBLE_DEVICES=visible, dashboard_address=":0")
    from distributed import LocalCluster

    return LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True, dashboard_address=":0")


def scan_distributed(
    fasta: str | Path,
    guide: str,
    max_mm: int = 4,
    backend: str = "cpu",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    main_chroms_only: bool = False,
    n_workers: int = 2,
    max_in_flight: int | None = None,
    on_chunk: Callable[[ChunkResult], None] | None = None,
    cluster=None,
) -> tuple[list[Hit], list[ChunkResult]]:
    """Same answer as ``pipeline.scan``, computed by a pool of Dask workers.

    Returns (final hits, one ChunkResult per chunk in the order they finished).
    Pass an existing ``cluster`` to reuse it (tests do this to save start-up time).
    """
    from distributed import Client, as_completed

    guide_enc = validate_guide(guide)
    get_backend(backend)  # fail early (e.g. no GPU) before starting workers
    max_in_flight = max_in_flight or 2 * n_workers  # keep every worker busy, never more

    own_cluster = cluster is None
    if own_cluster:
        cluster = make_cluster(backend, n_workers)
    try:
        with Client(cluster) as client:
            client.wait_for_workers(n_workers if own_cluster else 1)
            workers = sorted(client.scheduler_info()["workers"])
            # We pick the worker for each chunk ourselves.  Left to itself, Dask
            # sends almost every chunk to the same worker whenever searching is
            # faster than reading the next chunk (both workers look idle, and
            # the tie goes to the same one every time).
            in_flight = {w: 0 for w in workers}
            assigned_bases = {w: 0 for w in workers}
            target_of: dict[str, str] = {}  # future key -> worker address

            chunks = iter_chunks(fasta, chunk_size=chunk_size, main_chroms_only=main_chroms_only)
            pending = as_completed()
            next_index = 0

            def submit_next() -> bool:
                nonlocal next_index
                ch = next(chunks, None)
                if ch is None:
                    return False
                # least busy worker first, then the one that has had the fewest bases
                w = min(workers, key=lambda a: (in_flight[a], assigned_bases[a]))
                fut = client.submit(
                    search_task, next_index, ch.chrom, ch.offset, ch.seq, guide_enc, max_mm, backend,
                    pure=False, workers=[w], allow_other_workers=True,  # fall back if w dies
                )
                in_flight[w] += 1
                assigned_bases[w] += len(ch.seq)
                target_of[fut.key] = w
                pending.add(fut)
                next_index += 1
                return True

            for _ in range(max_in_flight):
                if not submit_next():
                    break

            results: list[ChunkResult] = []
            for fut in pending:  # yields futures as they finish
                res: ChunkResult = fut.result()  # re-raises a worker's error here
                in_flight[target_of.pop(fut.key)] -= 1
                fut.release()
                results.append(res)
                if on_chunk is not None:
                    on_chunk(res)
                submit_next()  # refill: one finished, one more goes out
    finally:
        if own_cluster:
            cluster.close()

    all_hits = [h for r in results for h in r.hits]
    return sort_hits(dedupe_hits(all_hits)), results


def work_share(results: list[ChunkResult]) -> dict[str, dict]:
    """Bases, chunks and busy time per device, plus each one's share of the bases."""
    per: dict[str, dict] = defaultdict(lambda: {"chunks": 0, "bases": 0, "seconds": 0.0})
    for r in results:
        d = per[f"{r.device} ({r.worker})"]
        d["chunks"] += 1
        d["bases"] += r.bases
        d["seconds"] += r.seconds
    total = sum(d["bases"] for d in per.values()) or 1
    for d in per.values():
        d["share"] = d["bases"] / total
    return dict(sorted(per.items()))


class WorkerLog:
    """TSV log: which worker/GPU processed each chunk, how many bases, how long."""

    COLUMNS = ["chunk", "chrom", "offset", "bases", "hits", "seconds", "worker", "device"]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w")
        self._fh.write("\t".join(self.COLUMNS) + "\n")

    def __call__(self, r: ChunkResult) -> None:
        row = [r.index, r.chrom, r.offset, r.bases, len(r.hits), f"{r.seconds:.4f}", r.worker, r.device]
        self._fh.write("\t".join(map(str, row)) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
