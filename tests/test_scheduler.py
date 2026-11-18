"""Dask scheduling on a CPU LocalCluster (runs on the Mac).

On a GPU machine the same code uses LocalCUDACluster; the GPU-specific test
at the bottom runs there (or on the simulator with NUMBA_ENABLE_CUDASIM=1).
"""

import pytest
from distributed import LocalCluster
from numba import cuda

from geneweaver.pipeline import scan
from geneweaver.scheduler import WorkerLog, scan_distributed, work_share

from test_cpu_baseline import EXPECTED, GUIDE, MOCK, as_tuples


@pytest.fixture(scope="module")
def cluster():
    # One cluster shared by all tests in this file: starting one takes ~1 s.
    with LocalCluster(n_workers=2, threads_per_worker=1, processes=True, dashboard_address=":0") as c:
        yield c


@pytest.mark.parametrize("backend", ["cpu", "numba-cpu"])
@pytest.mark.parametrize("chunk_size", [50, 100, 1000])
def test_distributed_equals_serial(cluster, backend, chunk_size):
    hits, results = scan_distributed(MOCK, GUIDE, max_mm=4, backend=backend, chunk_size=chunk_size, cluster=cluster)
    assert as_tuples(hits) == EXPECTED
    assert hits == scan(MOCK, GUIDE, max_mm=4, backend=backend, chunk_size=chunk_size)
    # every chunk processed exactly once
    assert sorted(r.index for r in results) == list(range(len(results)))


def test_both_workers_get_work(cluster):
    # Plenty of small chunks, so both workers must be used.
    _, results = scan_distributed(MOCK, GUIDE, backend="numba-cpu", chunk_size=30, cluster=cluster)
    share = work_share(results)
    assert len(share) == 2
    assert all(d["chunks"] > 0 for d in share.values())
    assert sum(d["bases"] for d in share.values()) == sum(r.bases for r in results)
    assert abs(sum(d["share"] for d in share.values()) - 1) < 1e-9


def test_in_flight_limit_and_callback(cluster):
    seen = []
    hits, results = scan_distributed(
        MOCK, GUIDE, backend="cpu", chunk_size=60, max_in_flight=1, on_chunk=seen.append, cluster=cluster
    )
    assert seen == results
    assert as_tuples(hits) == EXPECTED
    # with only one chunk in flight, results arrive strictly in order
    assert [r.index for r in results] == list(range(len(results)))


def test_worker_errors_are_raised(cluster):
    with pytest.raises(FileNotFoundError):
        scan_distributed("does/not/exist.fa", GUIDE, cluster=cluster)


def test_worker_log(cluster, tmp_path):
    log = WorkerLog(tmp_path / "workers.tsv")
    _, results = scan_distributed(MOCK, GUIDE, backend="numba-cpu", chunk_size=100, on_chunk=log, cluster=cluster)
    log.close()
    lines = (tmp_path / "workers.tsv").read_text().splitlines()
    assert lines[0].split("\t") == WorkerLog.COLUMNS
    assert len(lines) == 1 + len(results)
    assert {l.split("\t")[7] for l in lines[1:]} == {"CPU"}


def test_own_cluster_is_started_and_closed():
    hits, _ = scan_distributed(MOCK, GUIDE, backend="numba-cpu", chunk_size=100, n_workers=1)
    assert as_tuples(hits) == EXPECTED


@pytest.mark.skipif(not cuda.is_available(), reason="needs a GPU or NUMBA_ENABLE_CUDASIM=1")
def test_gpu_backend_through_dask(cluster):
    # On the simulator the worker processes inherit NUMBA_ENABLE_CUDASIM=1.
    hits, results = scan_distributed(MOCK, GUIDE, backend="gpu", chunk_size=100, cluster=cluster)
    assert as_tuples(hits) == EXPECTED
    assert all(r.device.startswith("GPU") for r in results)
