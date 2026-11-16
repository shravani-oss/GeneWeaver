"""Drive the dashboard headlessly with Textual's test pilot."""

import asyncio

from geneweaver.tui import NO_GPU_TEXT, GeneWeaverApp, GpuMonitor, ScanConfig, format_gpu_stats

from test_cpu_baseline import EXPECTED, GUIDE, MOCK


class NoGpu(GpuMonitor):
    def __init__(self):
        self._nvml, self._handles = None, []


async def run_app(config, size=(120, 40)):
    app = GeneWeaverApp(config, gpu_monitor=NoGpu())
    async with app.run_test(size=size) as pilot:
        for _ in range(200):  # wait up to ~10 s for the background scan
            if app.finished:
                break
            await pilot.pause(0.05)
        table = app.query_one("#hits")
        status = str(app.query_one("#status").render())
        gpu_text = str(app.query_one("#gpu").render())
        progress = app.query_one("#progress")
        return app, table.row_count, status, gpu_text, progress.progress, progress.total


def test_tui_cpu_scan_shows_progress_and_hits(tmp_path):
    out = tmp_path / "hits.tsv"
    config = ScanConfig(str(MOCK), GUIDE, max_mm=4, backend="cpu", chunk_size=100, out=str(out))
    app, rows, status, gpu_text, done, total = asyncio.run(run_app(config))
    assert app.error is None
    assert app.finished
    assert rows == len(EXPECTED)
    assert [(h.chrom, h.start, h.strand) for h in app.hits] == [e[:3] for e in EXPECTED]
    assert total == app.chunks_done == done
    assert total > 1
    assert "Done: 8 sites" in status
    assert gpu_text.strip() == NO_GPU_TEXT
    assert out.exists()


def test_tui_shows_errors_instead_of_crashing():
    config = ScanConfig("does/not/exist.fa", GUIDE)
    app, rows, status, *_ = asyncio.run(run_app(config))
    assert app.error is not None and "FileNotFoundError" in app.error
    assert "Error" in status
    assert rows == 0


def test_format_gpu_stats():
    assert format_gpu_stats([]) == NO_GPU_TEXT
    text = format_gpu_stats([{"index": 0, "name": "Tesla T4", "used": 2e9, "total": 16e9, "util": 97}])
    assert "GPU 0  Tesla T4" in text and "2.0 / 16.0 GB" in text and "97 %" in text
