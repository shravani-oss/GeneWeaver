"""Live terminal dashboard (Textual).

    ┌ GeneWeaver ─────────────────────────────────────────────────┐
    │ ┌ Progress ───────────────────────┐ ┌ GPU ────────────────┐ │
    │ │ chr21:5,000,000-10,000,000      │ │ GPU 0 Tesla T4      │ │
    │ │ ███████████░░░░░░  4/10  40%    │ │  mem 1.2/15.0 GB    │ │
    │ └─────────────────────────────────┘ │  util 97%           │ │
    │ ┌ Hits ────────────────────────────────────────────────────┐ │
    │ │ chrom  start  strand  site  mm  positions  label          │ │
    │ └──────────────────────────────────────────────────────────┘ │
    └──────────────────────────────────────────────────────────────┘

How it stays responsive: a terminal UI redraws itself in an event loop.
If the scan ran in that loop, the screen would freeze until it finished.
So the scan runs in a background *thread* (``@work(thread=True)``), and after
each chunk it asks the UI thread to update the widgets via
``call_from_thread``.  The GPU panel refreshes every second on a timer
(``set_interval``), independently of the scan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Label, ProgressBar, Static
from textual.worker import get_current_worker

from .parser import DEFAULT_CHUNK_SIZE, Chunk, count_chunks
from .pipeline import scan
from .results import Hit, write_tsv

NO_GPU_TEXT = "No NVIDIA GPU (CPU mode)"


@dataclass
class ScanConfig:
    fasta: str
    guide: str
    max_mm: int = 4
    backend: str = "numba-cpu"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    main_chroms_only: bool = False
    out: str | None = None


class ScanCancelled(Exception):
    pass


# ---------------------------------------------------------------------------
# GPU stats (pynvml), with a safe fallback when there is no NVIDIA GPU
# ---------------------------------------------------------------------------
class GpuMonitor:
    """Reads memory/utilization for every NVIDIA GPU via NVML.

    On the Mac (or anywhere without an NVIDIA driver) ``available`` is False
    and ``read()`` returns an empty list, instead of crashing.
    """

    def __init__(self) -> None:
        self._nvml = None
        self._handles: list = []
        try:
            import pynvml  # provided by the nvidia-ml-py package

            pynvml.nvmlInit()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
            self._nvml = pynvml
        except Exception:
            self._nvml = None
            self._handles = []

    @property
    def available(self) -> bool:
        return bool(self._handles)

    def read(self) -> list[dict]:
        if not self.available:
            return []
        nv = self._nvml
        stats = []
        for i, h in enumerate(self._handles):
            name = nv.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            util = nv.nvmlDeviceGetUtilizationRates(h)
            stats.append({"index": i, "name": name, "used": mem.used, "total": mem.total, "util": util.gpu})
        return stats

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


def format_gpu_stats(stats: list[dict]) -> str:
    if not stats:
        return NO_GPU_TEXT
    lines = []
    for s in stats:
        lines.append(f"GPU {s['index']}  {s['name']}")
        lines.append(f"  mem  {s['used'] / 1e9:5.1f} / {s['total'] / 1e9:.1f} GB")
        lines.append(f"  util {s['util']:3d} %")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------
HIT_COLUMNS = ("chrom", "start", "strand", "site", "mm", "positions", "label")


def hit_row(h: Hit) -> tuple:
    positions = ",".join(map(str, h.mismatch_positions)) or "."
    return (h.chrom, f"{h.start:,}", h.strand, h.site, h.mismatches, positions, h.label)


class GeneWeaverApp(App):
    TITLE = "GeneWeaver"
    CSS = """
    #top { height: auto; }
    #progress-box { width: 1fr; height: auto; border: round $accent; padding: 0 1; }
    #gpu { width: 36; height: auto; min-height: 5; border: round $accent; padding: 0 1; }
    #hits { height: 1fr; border: round $accent; }
    ProgressBar { margin-top: 1; }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, config: ScanConfig, gpu_monitor: GpuMonitor | None = None) -> None:
        super().__init__()
        self.config = config
        self.gpu = gpu_monitor if gpu_monitor is not None else GpuMonitor()
        self.hits: list[Hit] = []
        self.chunks_done = 0
        self.total_chunks: int | None = None
        self.finished = False
        self.error: str | None = None

    # ----- layout -------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top"):
            with Vertical(id="progress-box"):
                yield Label(self._header_text(), id="config")
                yield Label("Counting chunks…", id="status")
                yield ProgressBar(id="progress", show_eta=True)
            yield Static(NO_GPU_TEXT, id="gpu")
        yield DataTable(id="hits", zebra_stripes=True)
        yield Footer()

    def _header_text(self) -> str:
        c = self.config
        return f"{Path(c.fasta).name}  guide {c.guide}  ≤{c.max_mm} mm  backend {c.backend}"

    def on_mount(self) -> None:
        self.query_one("#progress-box").border_title = "Progress"
        self.query_one("#gpu").border_title = "GPU"
        table = self.query_one("#hits", DataTable)
        table.border_title = "Hits"
        table.add_columns(*HIT_COLUMNS)
        self.refresh_gpu()
        self.set_interval(1.0, self.refresh_gpu)
        self.run_scan()

    def on_unmount(self) -> None:
        self.gpu.close()

    # ----- GPU panel (timer, UI thread) ---------------------------------
    def refresh_gpu(self) -> None:
        self.query_one("#gpu", Static).update(format_gpu_stats(self.gpu.read()))

    # ----- the scan (background thread) ---------------------------------
    @work(thread=True, exclusive=True)
    def run_scan(self) -> None:
        worker = get_current_worker()
        c = self.config
        try:
            total = count_chunks(c.fasta, c.chunk_size, c.main_chroms_only)
            self.call_from_thread(self._set_total, total)

            def on_chunk(chunk: Chunk, chunk_hits: list[Hit]) -> None:
                if worker.is_cancelled:
                    raise ScanCancelled()
                self.call_from_thread(self._chunk_done, chunk, chunk_hits)

            t0 = time.perf_counter()
            hits = scan(
                c.fasta, c.guide, max_mm=c.max_mm, backend=c.backend,
                chunk_size=c.chunk_size, main_chroms_only=c.main_chroms_only, on_chunk=on_chunk,
            )
            elapsed = time.perf_counter() - t0
            if c.out:
                write_tsv(hits, c.out)
            self.call_from_thread(self._scan_finished, hits, elapsed)
        except ScanCancelled:
            pass
        except Exception as e:  # show the problem instead of crashing the UI
            self.call_from_thread(self._scan_failed, f"{type(e).__name__}: {e}")

    # ----- UI updates (always run on the UI thread) ---------------------
    def _set_total(self, total: int) -> None:
        self.total_chunks = total
        self.query_one("#progress", ProgressBar).update(total=total, progress=0)
        self.query_one("#status", Label).update(f"0 / {total} chunks")

    def _chunk_done(self, chunk: Chunk, chunk_hits: list[Hit]) -> None:
        self.chunks_done += 1
        self.query_one("#progress", ProgressBar).advance(1)
        self.query_one("#status", Label).update(
            f"{self.chunks_done} / {self.total_chunks} chunks   "
            f"{chunk.chrom}:{chunk.offset:,}-{chunk.end:,}   {len(chunk_hits)} hits"
        )
        table = self.query_one("#hits", DataTable)
        for h in chunk_hits:
            table.add_row(*hit_row(h))

    def _scan_finished(self, hits: list[Hit], elapsed: float) -> None:
        # Replace the live rows with the final de-duplicated, sorted list.
        self.hits = hits
        self.finished = True
        table = self.query_one("#hits", DataTable)
        table.clear()
        for h in hits:
            table.add_row(*hit_row(h))
        saved = f"  → {self.config.out}" if self.config.out else ""
        self.query_one("#status", Label).update(
            f"Done: {len(hits)} sites in {elapsed:.2f} s ({self.chunks_done} chunks){saved}   [q] quit"
        )

    def _scan_failed(self, message: str) -> None:
        self.error = message
        self.finished = True
        self.query_one("#status", Label).update(f"[b red]Error:[/] {message}")


def run_tui(config: ScanConfig) -> list[Hit]:
    app = GeneWeaverApp(config)
    app.run()
    if app.error:
        raise RuntimeError(app.error)
    return app.hits
