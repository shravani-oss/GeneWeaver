"""GPU results must exactly equal CPU results.

* Mac:    NUMBA_ENABLE_CUDASIM=1 pytest tests/test_gpu_matches_cpu.py -q
          (simulator; tiny inputs only; the chr21 test is skipped)
* Kaggle: pytest tests/test_gpu_matches_cpu.py -q
          (real GPU; also runs the chr21 test if data/chr21.fa exists)

Without a GPU or the simulator, every test here is skipped.
"""

import random
from pathlib import Path

import numpy as np
import pytest
from numba import cuda

from geneweaver.cpu_baseline import search_chunk_numba, search_chunk_python
from geneweaver.encoding import encode, validate_guide
from geneweaver.pipeline import scan

from test_cpu_baseline import EXPECTED, GUIDE, MOCK, as_tuples, hit_rich_sequence

pytestmark = pytest.mark.skipif(
    not cuda.is_available(), reason="no CUDA GPU (set NUMBA_ENABLE_CUDASIM=1 to use the simulator)"
)

from geneweaver.gpu_kernels import GpuSearcher, using_simulator  # noqa: E402

CHR21 = Path(__file__).resolve().parents[1] / "data" / "chr21.fa"


@pytest.mark.parametrize("chunk_size", [50, 100, 1_000_000])
def test_gpu_finds_planted_sites(chunk_size):
    hits = scan(MOCK, GUIDE, max_mm=4, backend="gpu", chunk_size=chunk_size)
    assert as_tuples(hits) == EXPECTED


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("max_mm", [0, 4, 6])
def test_gpu_matches_cpu_on_random_sequences(seed, max_mm):
    seq = encode(hit_rich_sequence(random.Random(seed), 40))
    guide = validate_guide(GUIDE)
    expected = search_chunk_python(seq, guide, max_mm)
    assert len(expected) > 0
    assert GpuSearcher(guide).search(seq, max_mm) == expected


def test_buffers_are_reused_and_stale_data_cannot_leak():
    """A big chunk, then a small one: buffers are not reallocated, and the
    small chunk's results contain nothing left over from the big chunk."""
    guide = validate_guide(GUIDE)
    rng = random.Random(7)
    big = encode(hit_rich_sequence(rng, 30))
    small = encode(hit_rich_sequence(rng, 5))
    assert len(small) < len(big)

    s = GpuSearcher(guide)
    assert s.search(big, 6) == search_chunk_numba(big, guide, 6)
    assert s.search(small, 6) == search_chunk_numba(small, guide, 6)
    assert s.search(big, 6) == search_chunk_numba(big, guide, 6)
    assert s.allocations == 1


def test_count_arrays_match_numba_exactly():
    """Not just the hits: every per-position count (incl. 255) must agree."""
    from geneweaver.cpu_baseline import count_mismatches_numba
    from geneweaver.encoding import reverse_complement

    guide = validate_guide(GUIDE)
    seq = encode(hit_rich_sequence(random.Random(3), 20))
    gp, gm = GpuSearcher(guide).count(seq)
    cp, cm = count_mismatches_numba(seq, guide, np.ascontiguousarray(reverse_complement(guide)))
    np.testing.assert_array_equal(gp, cp)
    np.testing.assert_array_equal(gm, cm)


@pytest.mark.parametrize("length", [0, 22, 23, 24])
def test_tiny_chunks(length):
    seq = encode((GUIDE + "AGGA")[:length])
    hits = GpuSearcher(validate_guide(GUIDE)).search(seq, 4)
    assert hits == ([(0, "+", 0)] if length >= 23 else [])


@pytest.mark.skipif(using_simulator(), reason="chr21 is far too big for the simulator")
@pytest.mark.skipif(not CHR21.exists(), reason="data/chr21.fa not downloaded")
@pytest.mark.parametrize("max_mm", [4, 6])
def test_gpu_matches_cpu_on_chr21(max_mm):
    kwargs = dict(max_mm=max_mm, chunk_size=5_000_000)
    gpu = scan(CHR21, GUIDE, backend="gpu", **kwargs)
    cpu = scan(CHR21, GUIDE, backend="numba-cpu", **kwargs)
    assert gpu == cpu
