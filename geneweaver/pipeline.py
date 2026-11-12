"""Glue: FASTA -> chunks -> search -> hits.

This is the simple one-process version.  Week 3 adds ``scheduler.py``, which
sends the same chunks to Dask workers (one per GPU) instead.
"""

from __future__ import annotations

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
