"""Summarize a --chunk-log file and check that GPU memory stayed flat.

Usage:
    python benchmarks/memory_report.py results/hg38_chunks.tsv

How the check works: the GPU buffers grow until they first see a full-size
chunk, then are reused.  So we take GPU memory use right after the first
full-size chunk as the baseline, and require that no later chunk uses more
than ``--tolerance-mb`` above it.  Growth beyond that means memory is leaking.
Exit code 0 = flat, 1 = growing (or no GPU data in the log).
"""

from __future__ import annotations

import argparse
import csv
import sys


def load(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def check(rows: list[dict], tolerance_mb: float = 64.0) -> tuple[bool, list[str]]:
    lines: list[str] = []
    if not rows:
        return False, ["log is empty"]

    bases = [int(r["bases"]) for r in rows]
    seconds = [float(r["seconds"]) for r in rows]
    total_bases = sum(bases)
    total_s = sum(seconds)
    lines.append(f"chunks:        {len(rows)}")
    lines.append(f"bases:         {total_bases:,} (incl. 22-base overlaps)")
    lines.append(f"time:          {total_s:.1f} s  ({total_bases / total_s / 1e6:.1f} M bases/s overall)" if total_s else "time: 0 s")
    lines.append(f"slowest chunk: {max(seconds):.2f} s")

    if rows[0]["gpu_used_mb"] == ".":
        lines.append("no GPU memory columns (the run was not --backend gpu)")
        return False, lines

    used = [float(r["gpu_used_mb"]) for r in rows]
    total_mb = float(rows[0]["gpu_total_mb"])
    first_full = bases.index(max(bases))
    baseline = used[first_full]
    growth = max(used[first_full:]) - baseline
    flat = growth <= tolerance_mb

    lines.append(f"GPU total:     {total_mb:,.0f} MB")
    lines.append(f"GPU used:      min {min(used):,.0f} MB, max {max(used):,.0f} MB")
    lines.append(f"baseline:      {baseline:,.0f} MB after chunk {first_full} (first full-size chunk)")
    lines.append(f"growth after baseline: {growth:+,.1f} MB (tolerance {tolerance_mb:.0f} MB)")
    lines.append("verdict:       " + ("FLAT ✅ no leak" if flat else "GROWING ❌ possible leak"))
    return flat, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunk_log")
    ap.add_argument("--tolerance-mb", type=float, default=64.0)
    args = ap.parse_args(argv)
    ok, lines = check(load(args.chunk_log), args.tolerance_mb)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
