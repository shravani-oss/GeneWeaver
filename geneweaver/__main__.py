"""Command line:  python -m geneweaver scan --fasta ... --guide ..."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

from pathlib import Path

from .cpu_baseline import BACKENDS, get_backend
from .encoding import validate_guide
from .parser import DEFAULT_CHUNK_SIZE
from .pipeline import scan
from .results import write_tsv


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="geneweaver", description="CRISPR off-target search")
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="search a FASTA genome for off-target sites")
    s.add_argument("--fasta", required=True, help="genome FASTA (.fa or .fa.gz)")
    s.add_argument("--guide", required=True, help="20-base guide sequence (A/C/G/T)")
    s.add_argument("--max-mm", type=int, default=4, help="maximum mismatches (default 4)")
    s.add_argument("--backend", choices=BACKENDS, default="numba-cpu",
                   help="search engine (gpu needs an NVIDIA GPU or NUMBA_ENABLE_CUDASIM=1)")
    s.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="bases per chunk")
    s.add_argument("--main-chroms-only", action="store_true", help="keep only chr1-22, X, Y")
    s.add_argument("--out", default="results/hits.tsv", help="output TSV path")
    s.add_argument("--quiet", action="store_true", help="no per-chunk progress lines")
    s.add_argument("--tui", action="store_true", help="show the live terminal dashboard")
    return ap


def cmd_scan(args: argparse.Namespace) -> int:
    # Check inputs before starting, so mistakes show up as a plain error
    # message here rather than inside the dashboard.
    validate_guide(args.guide)
    get_backend(args.backend)
    if not Path(args.fasta).exists():
        raise FileNotFoundError(f"FASTA file not found: {args.fasta}")

    t0 = time.perf_counter()
    if args.tui:
        from .tui import ScanConfig, run_tui  # Textual is only imported when needed

        hits = run_tui(ScanConfig(
            fasta=args.fasta, guide=args.guide, max_mm=args.max_mm, backend=args.backend,
            chunk_size=args.chunk_size, main_chroms_only=args.main_chroms_only, out=args.out,
        ))
    else:
        def progress(chunk, hits):
            if not args.quiet:
                print(f"  {chunk.chrom}:{chunk.offset:,}-{chunk.end:,}  {len(hits)} hits", file=sys.stderr)

        hits = scan(
            args.fasta, args.guide, max_mm=args.max_mm, backend=args.backend,
            chunk_size=args.chunk_size, main_chroms_only=args.main_chroms_only, on_chunk=progress,
        )
        write_tsv(hits, args.out)
    dt = time.perf_counter() - t0

    by_mm = Counter(h.mismatches for h in hits)
    print(f"{len(hits)} sites in {dt:.2f} s ({args.backend}) -> {args.out}")
    for mm in range(args.max_mm + 1):
        print(f"  {mm} mismatches: {by_mm.get(mm, 0)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "scan":
            return cmd_scan(args)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
