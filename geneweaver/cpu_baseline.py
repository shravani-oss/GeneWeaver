"""CPU search for CRISPR off-target sites.

Two versions that must always give identical answers:

* ``search_chunk_python`` - a plain Python loop.  Slow, but simple enough to
  trust.  It is the "1x" reference that all speedups are measured against.
* ``search_chunk_numba``  - the same algorithm compiled by Numba (``@njit``)
  into machine code.  This is a fair "CPU at full speed" comparison for the GPU.

How a window is checked (window = 23 bases starting at position p):

  + strand:  seq[p .. p+19] vs guide,             PAM seq[p+21], seq[p+22] == G G
  - strand:  seq[p] seq[p+1] == C C (the PAM "NGG" seen from the other strand),
             seq[p+3 .. p+22] vs reverse_complement(guide)

So the minus strand is searched without ever building a reverse-complement
copy of the genome.  ``N`` in the genome never equals a guide base, so it
automatically counts as a mismatch, and an ``N`` in a required PAM G/C makes
the window invalid.

Both functions return a list of ``(local_pos, strand, mismatches)`` sorted by
position, then strand ('+' before '-').
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numba import njit

from .encoding import C, G, GUIDE_LEN, PAM_LEN, SITE_LEN, reverse_complement

# Marker for "no valid PAM here" in the per-position count arrays.
INVALID = 255

RawHit = tuple[int, str, int]  # (local position, strand, mismatch count)


def _check_max_mm(max_mm: int) -> None:
    if not 0 <= max_mm <= GUIDE_LEN:
        raise ValueError(f"max_mm must be between 0 and {GUIDE_LEN}, got {max_mm}")


# ---------------------------------------------------------------------------
# Pure Python
# ---------------------------------------------------------------------------
def search_chunk_python(seq: np.ndarray, guide: np.ndarray, max_mm: int) -> list[RawHit]:
    """Check every position of ``seq`` with a plain Python loop."""
    _check_max_mm(max_mm)
    # Indexing a bytes object gives Python ints and is much faster than
    # indexing a NumPy array element by element.
    s = seq.tobytes()
    g = guide.tolist()
    rc = reverse_complement(guide).tolist()
    n_windows = len(s) - SITE_LEN + 1

    hits: list[RawHit] = []
    for p in range(max(n_windows, 0)):
        # + strand: guide then NGG
        if s[p + GUIDE_LEN + 1] == G and s[p + GUIDE_LEN + 2] == G:
            mm = 0
            for k in range(GUIDE_LEN):
                if s[p + k] != g[k]:
                    mm += 1
                    if mm > max_mm:
                        break  # already too many, stop counting
            if mm <= max_mm:
                hits.append((p, "+", mm))

        # - strand: CCN then reverse_complement(guide)
        if s[p] == C and s[p + 1] == C:
            mm = 0
            for k in range(GUIDE_LEN):
                if s[p + PAM_LEN + k] != rc[k]:
                    mm += 1
                    if mm > max_mm:
                        break
            if mm <= max_mm:
                hits.append((p, "-", mm))
    return hits


# ---------------------------------------------------------------------------
# Numba CPU
# ---------------------------------------------------------------------------
@njit(cache=True)
def count_mismatches_numba(seq, guide, rc_guide):
    """For every window start, write the mismatch count (or INVALID).

    Returns two uint8 arrays (+ strand, - strand) of length len(seq) - 22.
    This "one output slot per position" layout is exactly what the GPU kernel
    will produce, where each GPU thread fills in one slot.
    """
    n = seq.shape[0] - SITE_LEN + 1
    if n < 0:
        n = 0
    out_plus = np.full(n, INVALID, dtype=np.uint8)
    out_minus = np.full(n, INVALID, dtype=np.uint8)
    for p in range(n):
        if seq[p + GUIDE_LEN + 1] == G and seq[p + GUIDE_LEN + 2] == G:
            mm = 0
            for k in range(GUIDE_LEN):
                if seq[p + k] != guide[k]:
                    mm += 1
            out_plus[p] = mm
        if seq[p] == C and seq[p + 1] == C:
            mm = 0
            for k in range(GUIDE_LEN):
                if seq[p + PAM_LEN + k] != rc_guide[k]:
                    mm += 1
            out_minus[p] = mm
    return out_plus, out_minus


def hits_from_counts(out_plus: np.ndarray, out_minus: np.ndarray, max_mm: int) -> list[RawHit]:
    """Turn per-position count arrays into a sorted list of raw hits.

    Shared by the Numba CPU backend and (later) the GPU backend.
    INVALID (255) is always > max_mm, so invalid windows drop out here.
    """
    hits: list[RawHit] = [(int(p), "+", int(out_plus[p])) for p in np.nonzero(out_plus <= max_mm)[0]]
    hits += [(int(p), "-", int(out_minus[p])) for p in np.nonzero(out_minus <= max_mm)[0]]
    hits.sort()
    return hits


def search_chunk_numba(seq: np.ndarray, guide: np.ndarray, max_mm: int) -> list[RawHit]:
    """Same result as ``search_chunk_python``, using compiled Numba code."""
    _check_max_mm(max_mm)
    seq = np.ascontiguousarray(seq, dtype=np.uint8)
    guide = np.ascontiguousarray(guide, dtype=np.uint8)
    rc = np.ascontiguousarray(reverse_complement(guide))
    out_plus, out_minus = count_mismatches_numba(seq, guide, rc)
    return hits_from_counts(out_plus, out_minus, max_mm)


SearchFn = Callable[[np.ndarray, np.ndarray, int], list[RawHit]]

BACKENDS: dict[str, SearchFn] = {
    "cpu": search_chunk_python,
    "numba-cpu": search_chunk_numba,
}


def get_backend(name: str) -> SearchFn:
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}") from None
