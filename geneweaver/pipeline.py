"""Glue: FASTA -> chunks -> search -> hits.

This is the simple one-process version.  Week 3 adds ``scheduler.py``, which
sends the same chunks to Dask workers (one per GPU) instead.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .cpu_baseline import get_backend
from .encoding import validate_guide
from .parser import DEFAULT_CHUNK_SIZE, Chunk, iter_chunks
from .results import Hit, dedupe_hits, make_hit, sort_hits

# Called after each chunk with (chunk, hits found in that chunk).
ProgressFn = Callable[[Chunk, list[Hit]], None]


def scan(
    fasta: str | Path,
    guide: str,
    max_mm: int = 4,
    backend: str = "cpu",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    main_chroms_only: bool = False,
    on_chunk: ProgressFn | None = None,
) -> list[Hit]:
    guide_enc = validate_guide(guide)
    search = get_backend(backend)

    all_hits: list[Hit] = []
    for chunk in iter_chunks(fasta, chunk_size=chunk_size, main_chroms_only=main_chroms_only):
        chunk_hits = [
            make_hit(chunk.chrom, chunk.offset, chunk.seq, pos, strand, mm, guide_enc)
            for pos, strand, mm in search(chunk.seq, guide_enc, max_mm)
        ]
        all_hits.extend(chunk_hits)
        if on_chunk is not None:
            on_chunk(chunk, chunk_hits)

    return sort_hits(dedupe_hits(all_hits))


class ChunkLog:
    """Write one TSV row per finished chunk: what it was, how long it took,
    and (for GPU runs) how much GPU memory was in use afterwards.

    Use it as the ``on_chunk`` callback of ``scan``.  If GPU memory use keeps
    rising from chunk to chunk, something is leaking; it should stay flat.
    """

    COLUMNS = ["chunk", "chrom", "offset", "bases", "hits", "seconds", "gpu_used_mb", "gpu_free_mb", "gpu_total_mb"]

    def __init__(self, path: str | Path, memory_info: Callable[[], tuple[int, int]] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_info = memory_info  # returns (free_bytes, total_bytes)
        self._fh = open(self.path, "w")
        self._fh.write("\t".join(self.COLUMNS) + "\n")
        self._n = 0
        self._last = time.perf_counter()

    def __call__(self, chunk: Chunk, hits: list[Hit]) -> None:
        now = time.perf_counter()
        row = [self._n, chunk.chrom, chunk.offset, len(chunk.seq), len(hits), f"{now - self._last:.4f}"]
        if self.memory_info is not None:
            free, total = self.memory_info()
            row += [f"{(total - free) / 2**20:.1f}", f"{free / 2**20:.1f}", f"{total / 2**20:.1f}"]
        else:
            row += [".", ".", "."]
        self._fh.write("\t".join(map(str, row)) + "\n")
        self._fh.flush()  # so the log is readable even if the run crashes
        self._n += 1
        self._last = time.perf_counter()

    def close(self) -> None:
        self._fh.close()
