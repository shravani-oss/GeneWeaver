"""Benchmark + memory-logging tooling (used on Kaggle in the mid-project review)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from numba import cuda

from geneweaver.encoding import decode
from geneweaver.parser import read_region
from geneweaver.pipeline import ChunkLog, scan

from test_cpu_baseline import GUIDE, MOCK

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look the module up here
    spec.loader.exec_module(mod)
    return mod


def mock_records():
    recs, name = {}, None
    for line in MOCK.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            recs[name] = ""
        else:
            recs[name] += line.upper()
    return recs


# ----- read_region ---------------------------------------------------------
@pytest.mark.parametrize("chrom,start,length", [("chrA", 0, 400), ("chrA", 55, 123), ("chrB", 250, 50), (None, 3, 10)])
def test_read_region(chrom, start, length):
    recs = mock_records()
    expected = recs[chrom or "chrA"][start : start + length]
    expected = "".join(b if b in "ACGT" else "N" for b in expected)
    assert decode(read_region(MOCK, chrom, start, length)) == expected


def test_read_region_out_of_range():
    with pytest.raises(ValueError):
        read_region(MOCK, "chrB", 290, 20)
    with pytest.raises(ValueError):
        read_region(MOCK, "chrZ", 0, 5)


# ----- ChunkLog + memory report ---------------------------------------------
def test_chunk_log_without_gpu(tmp_path):
    log = ChunkLog(tmp_path / "log.tsv")
    scan(MOCK, GUIDE, chunk_size=100, on_chunk=log)
    log.close()
    lines = (tmp_path / "log.tsv").read_text().splitlines()
    assert lines[0].split("\t") == ChunkLog.COLUMNS
    rows = [l.split("\t") for l in lines[1:]]
    assert [r[0] for r in rows] == [str(i) for i in range(len(rows))]
    assert sum(int(r[4]) for r in rows) == 8  # hits
    assert all(r[6:] == [".", ".", "."] for r in rows)


def test_chunk_log_with_memory(tmp_path):
    # (free, total) in MB: flat apart from a little jitter, as on a healthy GPU
    readings = iter([(800, 1000), (790, 1000), (805, 1000)] + [(800, 1000)] * 50)
    log = ChunkLog(tmp_path / "log.tsv", memory_info=lambda: tuple(x * 2**20 for x in next(readings)))
    scan(MOCK, GUIDE, chunk_size=100, on_chunk=log)
    log.close()
    report = load_script("memory_report")
    ok, lines = report.check(report.load(tmp_path / "log.tsv"))
    assert ok, lines
    assert any("FLAT" in l for l in lines)


def test_memory_report_detects_growth():
    report = load_script("memory_report")
    rows = [
        {"bases": "100", "seconds": "1", "gpu_used_mb": str(500 + 100 * i), "gpu_free_mb": "0", "gpu_total_mb": "16000"}
        for i in range(5)
    ]
    ok, lines = report.check(rows)
    assert not ok and any("GROWING" in l for l in lines)


def test_cli_chunk_log(tmp_path):
    out, log = tmp_path / "hits.tsv", tmp_path / "chunks.tsv"
    r = subprocess.run(
        [sys.executable, "-m", "geneweaver", "scan", "--fasta", str(MOCK), "--guide", GUIDE,
         "--backend", "numba-cpu", "--chunk-size", "100", "--out", str(out), "--chunk-log", str(log), "--quiet"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    assert len(log.read_text().splitlines()) > 2


# ----- bench.py ---------------------------------------------------------------
def run_bench(tmp_path, *extra):
    bench = load_script("bench")
    out = tmp_path / "results.md"
    code = bench.main([str(MOCK), "--chrom", "chrA", "--slice-start", "0", "--slice-len", "400",
                       "--repeats", "1", "--out", str(out), *extra])
    return code, out.read_text()


def test_bench_cpu(tmp_path):
    code, md = run_bench(tmp_path, "--backends", "cpu,numba-cpu", "--full", str(MOCK),
                         "--full-backends", "cpu,numba-cpu")
    assert code == 0
    assert "| Pure Python (CPU) |" in md and "| 1× |" in md
    assert "Numba CPU (1 core)" in md
    assert "❌" not in md and md.count("✅") == 3


@pytest.mark.skipif(not cuda.is_available(), reason="needs a GPU or NUMBA_ENABLE_CUDASIM=1")
def test_bench_gpu_code_path(tmp_path):
    code, md = run_bench(tmp_path, "--allow-simulator", "--full", str(MOCK), "--full-backends", "numba-cpu,gpu")
    assert code == 0, md
    assert "GPU kernel only" in md and "GPU end-to-end" in md and "GPU, 1 device (pipeline)" in md
    assert "❌" not in md
