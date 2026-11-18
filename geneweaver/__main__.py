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
from .pipeline import ChunkLog, scan
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
    s.add_argument("--chunk-log", metavar="TSV",
                   help="write per-chunk timing (and GPU memory for --backend gpu) to this file")
    s.add_argument("--gpus", type=int, default=0,
                   help="spread chunks over N GPUs with Dask (needs --backend gpu)")
    s.add_argument("--workers", type=int, default=0,
                   help="spread chunks over N CPU worker processes with Dask (cpu / numba-cpu backends)")
    return ap


def run_distributed(args: argparse.Namespace, n_workers: int):
    from .scheduler import WorkerLog, scan_distributed, work_share

    log = WorkerLog(args.chunk_log) if args.chunk_log else None

    def progress(r):
        if log is not None:
            log(r)
        if not args.quiet:
            print(f"  {r.chrom}:{r.offset:,}-{r.offset + r.bases:,}  {len(r.hits)} hits  "
                  f"[{r.device}, {r.worker}, {r.seconds:.2f} s]", file=sys.stderr)

    try:
        hits, results = scan_distributed(
            args.fasta, args.guide, max_mm=args.max_mm, backend=args.backend, chunk_size=args.chunk_size,
            main_chroms_only=args.main_chroms_only, n_workers=n_workers, on_chunk=progress,
        )
    finally:
        if log is not None:
            log.close()

    print("work per worker:")
    for name, d in work_share(results).items():
        print(f"  {name}: {d['chunks']} chunks, {d['bases']:,} bases ({d['share']:.1%}), busy {d['seconds']:.2f} s")
    return hits


def cmd_scan(args: argparse.Namespace) -> int:
    # Check inputs before starting, so mistakes show up as a plain error
    # message here rather than inside the dashboard.
    validate_guide(args.guide)
    get_backend(args.backend)
    if not Path(args.fasta).exists():
        raise FileNotFoundError(f"FASTA file not found: {args.fasta}")
    if args.tui and args.chunk_log:
        raise ValueError("--chunk-log cannot be combined with --tui yet")
    if args.gpus and args.backend != "gpu":
        raise ValueError("--gpus needs --backend gpu (use --workers for CPU backends)")
    if args.workers and args.backend == "gpu":
        raise ValueError("--workers is for CPU backends; use --gpus with --backend gpu")
    n_workers = args.gpus or args.workers
    if n_workers < 0:
        raise ValueError("--gpus / --workers must be positive")
    if n_workers and args.tui:
        raise ValueError("--tui with --gpus/--workers arrives in Week 4")

    t0 = time.perf_counter()
    if n_workers:
        hits = run_distributed(args, n_workers)
        write_tsv(hits, args.out)
    elif args.tui:
        from .tui import ScanConfig, run_tui  # Textual is only imported when needed

        hits = run_tui(ScanConfig(
            fasta=args.fasta, guide=args.guide, max_mm=args.max_mm, backend=args.backend,
            chunk_size=args.chunk_size, main_chroms_only=args.main_chroms_only, out=args.out,
        ))
    else:
        log = None
        if args.chunk_log:
            memory_info = None
            if args.backend == "gpu":
                from .gpu_kernels import gpu_memory_info

                memory_info = gpu_memory_info
            log = ChunkLog(args.chunk_log, memory_info)

        def progress(chunk, hits):
            if log is not None:
                log(chunk, hits)
            if not args.quiet:
                print(f"  {chunk.chrom}:{chunk.offset:,}-{chunk.end:,}  {len(hits)} hits", file=sys.stderr)

        try:
            hits = scan(
                args.fasta, args.guide, max_mm=args.max_mm, backend=args.backend,
                chunk_size=args.chunk_size, main_chroms_only=args.main_chroms_only, on_chunk=progress,
            )
        finally:
            if log is not None:
                log.close()
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
