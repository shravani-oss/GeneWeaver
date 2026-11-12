"""Hit records, de-duplication and TSV output.

Conventions (used everywhere in GeneWeaver):

* ``start``  - 0-based position of the leftmost base of the 23-base window on
  the forward (+) strand, for both strands.
* ``site``   - the 23 bases read in the guide's direction (guide part + PAM).
  For '-' hits this is the reverse complement of the forward-strand window,
  so every site looks like ``<20 bases>NGG``.  Mismatched bases are lowercase.
* ``mismatch_positions`` - 1-based positions in the guide (1 = the 5' end,
  farthest from the PAM; 20 = right next to the PAM).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .encoding import GUIDE_LEN, SITE_LEN, decode, reverse_complement

TSV_COLUMNS = ["chrom", "start", "strand", "site", "mismatches", "mismatch_positions", "score", "label"]
MISSING = "."


@dataclass(frozen=True)
class Hit:
    chrom: str
    start: int
    strand: str
    site: str
    mismatches: int
    mismatch_positions: tuple[int, ...]
    score: float | None = None

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chrom, self.start, self.strand)

    @property
    def label(self) -> str:
        return "on-target" if self.mismatches == 0 else "off-target"


def make_hit(
    chrom: str,
    chunk_offset: int,
    seq: np.ndarray,
    local_pos: int,
    strand: str,
    mismatches: int,
    guide: np.ndarray,
) -> Hit:
    """Build a full Hit record from a raw (position, strand, count) search result.

    The mismatches are recomputed here from the sequence, and we check they
    agree with what the search backend reported.  That catches backend bugs.
    """
    window = seq[local_pos : local_pos + SITE_LEN]
    if len(window) != SITE_LEN:
        raise ValueError(f"window at {chrom}:{chunk_offset + local_pos} runs past the chunk end")
    target = window if strand == "+" else reverse_complement(window)

    letters = list(decode(target))
    positions = []
    for k in range(GUIDE_LEN):
        if target[k] != guide[k]:
            positions.append(k + 1)
            letters[k] = letters[k].lower()

    if len(positions) != mismatches:
        raise AssertionError(
            f"backend reported {mismatches} mismatches at {chrom}:{chunk_offset + local_pos}{strand}, "
            f"but the sequence has {len(positions)}"
        )
    return Hit(chrom, chunk_offset + local_pos, strand, "".join(letters), mismatches, tuple(positions))


def dedupe_hits(hits: Iterable[Hit]) -> list[Hit]:
    """Drop repeats of the same (chrom, start, strand), e.g. from chunk overlaps."""
    seen: dict[tuple[str, int, str], Hit] = {}
    for h in hits:
        seen.setdefault(h.key, h)
    return list(seen.values())


def sort_hits(hits: Iterable[Hit]) -> list[Hit]:
    """Fewest mismatches first, then by genome position.

    (Week 3 will switch this to ranking by MIT/CFD score.)
    """
    return sorted(hits, key=lambda h: (h.mismatches, h.chrom, h.start, h.strand))


def _fmt_positions(pos: tuple[int, ...]) -> str:
    return ",".join(map(str, pos)) if pos else MISSING


def write_tsv(hits: Iterable[Hit], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(TSV_COLUMNS)
        for h in hits:
            w.writerow([
                h.chrom,
                h.start,
                h.strand,
                h.site,
                h.mismatches,
                _fmt_positions(h.mismatch_positions),
                MISSING if h.score is None else f"{h.score:.4f}",
                h.label,
            ])
