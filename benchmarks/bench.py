"""CPU vs GPU benchmark.  Writes benchmarks/results.md.

Run on Kaggle (same machine, same input for every backend):

    python benchmarks/bench.py data/chr21.fa --full data/chr21.fa

Part 1, the slice: cuts a fixed slice out of the genome (default
chr21:20,000,000-25,000,000; the first ~5 Mb of chr21 is all N, which
would make a meaningless test) and times each backend on exactly that array:

  * Pure Python           search_chunk_python             (the 1x reference)
  * Numba CPU             compiled loop + hit extraction
  * GPU kernel only       just the kernel (data already on the GPU)
  * GPU end-to-end        copy in + kernel + copy out + hit extraction

Part 2, whole files (optional, --full): runs the real pipeline
(parse + search + dedupe) on whole files, so the numbers include reading the
FASTA.

Rules (README section 6):
  * the first GPU / Numba call compiles the code; it is run once as a warm-up,
    not counted, and its time is reported separately
  * cuda.synchronize() before stopping any GPU timer, because kernel launches
    return before the GPU has finished
  * time.perf_counter(), each measurement repeated (default 3), median reported
  * a speedup only counts if every backend returned exactly the same hits
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numba
import numpy as np

from geneweaver.cpu_baseline import count_mismatches_numba, get_backend, hits_from_counts, search_chunk_python
from geneweaver.encoding import N, reverse_complement, validate_guide
from geneweaver.parser import read_region, record_lengths
from geneweaver.pipeline import scan

DEFAULT_GUIDE = "GAGTCCGAGCAGAAGAAGAA"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@dataclass
class Row:
    input: str
    backend: str
    times: list[float] = field(default_factory=list)
    bases: int = 0
    note: str = ""
    reference: bool = False  # the pure-Python slice row = 1×

    @property
    def median(self) -> float:
        return statistics.median(self.times)

    @property
    def bases_per_s(self) -> float:
        return self.bases / self.median


def timed(fn: Callable[[], object], repeats: int) -> tuple[list[float], object]:
    times, result = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return times, result


def fmt_time(s: float) -> str:
    if s < 1e-3:
        return f"{s * 1e6:.0f} µs"
    if s < 1:
        return f"{s * 1e3:.1f} ms"
    return f"{s:.2f} s"


def fmt_rate(r: float) -> str:
    for unit, div in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if r >= div:
            return f"{r / div:.1f} {unit}"
    return f"{r:.0f}"


def cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown CPU"


def gpu_status(allow_simulator: bool = False) -> tuple[bool, str]:
    """(usable for timing?, description)."""
    try:
        from numba import config, cuda

        if config.ENABLE_CUDASIM:
            if allow_simulator:
                return True, "CUDA simulator (code-path test only, timings meaningless)"
            return False, "CUDA simulator (timings would be meaningless; GPU skipped)"
        if not cuda.is_available():
            return False, "no NVIDIA GPU (GPU rows skipped)"
        name = cuda.get_current_device().name
        name = name.decode() if isinstance(name, bytes) else name
        return True, f"{name} (×{len(cuda.gpus)} available, benchmark uses 1)"
    except Exception as e:
        return False, f"GPU unavailable: {e}"


# ---------------------------------------------------------------------------
# Part 1: fixed slice
# ---------------------------------------------------------------------------
def bench_slice(seq, guide, max_mm, repeats, label, backends, use_gpu, notes):
    """Returns (rows, results at max_mm, search functions for a later re-check)."""
    rows: list[Row] = []
    results: dict[str, list] = {}
    search_fns: dict[str, Callable[[int], list]] = {}
    n = len(seq)

    if "cpu" in backends:
        search_fns["Pure Python"] = lambda mm: search_chunk_python(seq, guide, mm)
        times, hits = timed(lambda: search_fns["Pure Python"](max_mm), repeats)
        rows.append(Row(label, "Pure Python (CPU)", times, n, reference=True))
        results["Pure Python"] = hits

    if "numba-cpu" in backends:
        rc = np.ascontiguousarray(reverse_complement(guide))
        search_fns["Numba CPU"] = lambda mm: hits_from_counts(*count_mismatches_numba(seq, guide, rc), mm)

        first, _ = timed(lambda: search_fns["Numba CPU"](max_mm), 1)  # compile (or load from cache)
        times, hits = timed(lambda: search_fns["Numba CPU"](max_mm), repeats)
        rows.append(Row(label, "Numba CPU (1 core)", times, n))
        results["Numba CPU"] = hits
        notes.append(f"Numba CPU first call (compile or cache load, not counted): {fmt_time(first[0])}")

    if "gpu" in backends and use_gpu:
        from numba import cuda

        from geneweaver.gpu_kernels import GpuSearcher

        searcher = GpuSearcher(guide)
        t0 = time.perf_counter()
        searcher.search(seq, max_mm)  # warm-up: JIT-compiles the kernel
        cuda.synchronize()
        notes.append(f"GPU first call (JIT compile + allocate + run, not counted): {fmt_time(time.perf_counter() - t0)}")

        n_valid = searcher.upload(seq)

        def kernel_only():
            searcher.launch(n_valid)
            cuda.synchronize()  # wait for the GPU before stopping the clock

        times, _ = timed(kernel_only, repeats)
        rows.append(Row(label, "GPU kernel only", times, n))

        def end_to_end():
            hits = searcher.search(seq, max_mm)
            cuda.synchronize()
            return hits

        times, hits = timed(end_to_end, repeats)
        rows.append(Row(label, "GPU end-to-end (copy in + kernel + copy out + hits)", times, n))
        results["GPU"] = hits
        search_fns["GPU"] = lambda mm: searcher.search(seq, mm)

    return rows, results, search_fns


# ---------------------------------------------------------------------------
# Part 2: whole files through the real pipeline
# ---------------------------------------------------------------------------
def bench_full(path, guide_str, max_mm, repeats, backends, use_gpu, chunk_size, main_chroms_only):
    lengths = record_lengths(path, main_chroms_only)
    n = sum(lengths.values())
    label = f"{Path(path).name} ({n / 1e6:,.0f} M)"
    names = {"cpu": "Pure Python (CPU)", "numba-cpu": "Numba CPU (1 core)", "gpu": "GPU, 1 device (pipeline)"}
    tiny = np.zeros(1000, dtype=np.uint8)
    rows, results = [], {}
    for backend in backends:
        if backend == "gpu" and not use_gpu:
            continue
        get_backend(backend)(tiny, validate_guide(guide_str), max_mm)  # warm-up: compile outside the timer
        kwargs = dict(max_mm=max_mm, backend=backend, chunk_size=chunk_size, main_chroms_only=main_chroms_only)
        times, hits = timed(lambda: scan(path, guide_str, **kwargs), repeats)
        rows.append(Row(label, names[backend], times, n, note="includes FASTA parsing"))
        results[names[backend]] = hits
    return rows, results


def check_equal(results: dict[str, list]) -> tuple[bool, str]:
    if len(results) < 2:
        return True, "only one backend ran; nothing to compare"
    names = list(results)
    ref = results[names[0]]
    bad = [k for k in names[1:] if results[k] != ref]
    if bad:
        return False, f"❌ MISMATCH: {', '.join(bad)} differ from {names[0]}. Speedups below do NOT count."
    return True, f"✅ all backends returned identical hits ({len(ref)} sites: {', '.join(names)})"


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def render(header: list[str], sections: list[tuple[str, list[Row], str]], ref_rate: float | None, notes, repeats) -> str:
    out = ["# Benchmark results", ""] + header + [""]
    out += [
        f"| Input | Backend | Time (median of {repeats}) | Bases/sec | Speedup vs pure Python |",
        "|---|---|---|---|---|",
    ]
    for _, rows, _ in sections:
        for r in rows:
            if r.reference:
                speed = "1×"
            elif ref_rate:
                ratio = r.bases_per_s / ref_rate
                speed = f"{ratio:,.0f}×" if ratio >= 10 else f"{ratio:.1f}×"
            else:
                speed = "-"
            note = f" ({r.note})" if r.note else ""
            out.append(f"| {r.input} | {r.backend}{note} | {fmt_time(r.median)} | {fmt_rate(r.bases_per_s)} | {speed} |")
    out.append("")
    for title, _, verdict in sections:
        out.append(f"- **Correctness, {title}:** {verdict}")
    out += [f"- {n}" for n in notes]
    out += [
        "- Speedup = this row's bases/sec ÷ pure-Python bases/sec on the slice. "
        "Rows marked *includes FASTA parsing* are limited by reading the file, not by the search.",
        f"- All times: `time.perf_counter()`, median of {repeats} runs; GPU timers stop after `cuda.synchronize()`.",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("fasta", help="FASTA to cut the benchmark slice from (e.g. data/chr21.fa)")
    ap.add_argument("--chrom", default=None, help="chromosome for the slice (default: first record)")
    ap.add_argument("--slice-start", type=int, default=20_000_000)
    ap.add_argument("--slice-len", type=int, default=5_000_000)
    ap.add_argument("--guide", default=DEFAULT_GUIDE)
    ap.add_argument("--max-mm", type=int, default=4)
    ap.add_argument("--check-mm", type=int, default=8,
                    help="extra untimed correctness check at this many mismatches")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--backends", default="cpu,numba-cpu,gpu", help="comma-separated, for the slice")
    ap.add_argument("--full", action="append", default=[], metavar="FASTA",
                    help="also time the whole pipeline on this file (repeatable)")
    ap.add_argument("--full-backends", default="numba-cpu,gpu")
    ap.add_argument("--full-repeats", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=50_000_000)
    ap.add_argument("--main-chroms-only", action="store_true", help="for --full files")
    ap.add_argument("--out", default="benchmarks/results.md")
    ap.add_argument("--allow-simulator", action="store_true",
                    help="run GPU rows on the CUDA simulator (tests only; tiny inputs)")
    args = ap.parse_args(argv)

    guide = validate_guide(args.guide)
    use_gpu, gpu_desc = gpu_status(args.allow_simulator)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    seq = read_region(args.fasta, args.chrom, args.slice_start, args.slice_len)
    chrom = args.chrom or next(iter(record_lengths(args.fasta)))
    n_frac = float(np.count_nonzero(seq == N)) / len(seq)
    slice_label = f"{chrom}:{args.slice_start / 1e6:g}-{(args.slice_start + args.slice_len) / 1e6:g} Mb"
    print(f"slice {slice_label}: {len(seq):,} bases, {n_frac:.1%} N", file=sys.stderr)

    notes: list[str] = []
    slice_rows, slice_results, search_fns = bench_slice(
        seq, guide, args.max_mm, args.repeats, slice_label, backends, use_gpu, notes
    )
    ok, verdict = check_equal(slice_results)
    # A slice can easily have 0 hits at max 4 mismatches, and "all backends
    # found nothing" proves little.  So also compare (untimed) at a looser
    # threshold, which yields real sites to compare.
    ok2, verdict2 = check_equal({name: fn(args.check_mm) for name, fn in search_fns.items()})
    sections = [
        (f"slice, max {args.max_mm} mm (timed)", slice_rows, verdict),
        (f"slice, max {args.check_mm} mm (extra check, untimed)", [], verdict2),
    ]
    all_ok = ok and ok2

    for path in args.full:
        print(f"full run: {path}", file=sys.stderr)
        fb = [b.strip() for b in args.full_backends.split(",") if b.strip()]
        rows, results = bench_full(path, args.guide, args.max_mm, args.full_repeats, fb, use_gpu,
                                   args.chunk_size, args.main_chroms_only)
        ok, verdict = check_equal(results)
        all_ok &= ok
        sections.append((Path(path).name, rows, verdict))

    ref = next((r.bases_per_s for r in slice_rows if r.reference), None)
    header = [
        f"_Generated by `benchmarks/bench.py` on {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC_",
        "",
        f"- **CPU:** {cpu_name()}",
        f"- **GPU:** {gpu_desc}",
        f"- **Software:** Python {platform.python_version()}, Numba {numba.__version__}, NumPy {np.__version__}",
        f"- **Slice:** {slice_label} ({len(seq):,} bases, {n_frac:.1%} N); guide `{args.guide}`; max {args.max_mm} mismatches",
    ]
    report = render(header, sections, ref, notes, args.repeats)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report)
    print(report)
    print(f"written to {args.out}", file=sys.stderr)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
