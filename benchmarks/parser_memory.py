"""Check that streaming a FASTA keeps memory flat.

Usage:
    python benchmarks/parser_memory.py data/chr21.fa --chunk-size 5000000

Prints the process memory (RSS) after each chunk.  If the parser were loading
the whole file, RSS would keep climbing; with streaming it should level off
after the first chunk or two.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time

from geneweaver.parser import iter_chunks


def rss_mb() -> float:
    """Current resident memory of this process in MB (works on macOS and Linux)."""
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True)
    return int(out.stdout.strip()) / 1024  # ps reports KB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta")
    ap.add_argument("--chunk-size", type=int, default=5_000_000)
    ap.add_argument("--main-chroms-only", action="store_true")
    args = ap.parse_args()

    start_rss = rss_mb()
    t0 = time.perf_counter()
    readings = []
    total = 0
    for i, ch in enumerate(iter_chunks(args.fasta, args.chunk_size, args.main_chroms_only)):
        total += len(ch.seq)
        r = rss_mb()
        readings.append(r)
        print(f"chunk {i:4d}  {ch.chrom}:{ch.offset:,}-{ch.end:,}  RSS {r:7.1f} MB")
    dt = time.perf_counter() - t0

    print(f"\nstart RSS {start_rss:.1f} MB, min {min(readings):.1f} MB, max {max(readings):.1f} MB")
    print(f"{len(readings)} chunks, {total:,} bases (incl. overlaps) in {dt:.1f} s")
    file_mb = os.path.getsize(args.fasta) / 1024**2
    print(f"file size {file_mb:.1f} MB; memory growth {max(readings) - start_rss:.1f} MB")


if __name__ == "__main__":
    main()
