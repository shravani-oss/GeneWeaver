"""Numba CUDA kernel: one GPU thread checks one genome position.

The big idea
------------
On the CPU we loop over positions one after another.  On the GPU we launch one
*thread* per position, and thousands of them run at the same time.  Each
thread does exactly what one loop iteration of ``count_mismatches_numba`` does:
check the PAM, count mismatches, and write one number into an output array.

    position:   0    1    2    3   ...  n_valid-1
    thread:     0    1    2    3   ...  n_valid-1
    out_plus:  [255, 3,  255, 255, ...]   (255 = no valid PAM here)

Threads are grouped into *blocks* of 256.  We launch
``ceil(n_valid / 256)`` blocks, so the last block may have extra threads with
nothing to do; the ``if i < n_valid`` guard makes them stop immediately.

Host vs device
--------------
The GPU (the "device") has its own memory.  Data must be copied there before
the kernel runs and copied back afterwards:

    chunk on CPU --to device--> kernel runs --copy to host--> counts on CPU

The CPU then picks the hits with ``np.nonzero(out <= max_mm)`` (shared with the
Numba CPU backend in ``hits_from_counts``).

Device buffers are allocated once and reused for every chunk, because
allocating GPU memory is slow and doing it per chunk can fragment memory.

On the Mac
----------
There is no NVIDIA GPU.  Run with ``NUMBA_ENABLE_CUDASIM=1`` to use Numba's
CUDA *simulator*, which runs the same kernel code on the CPU.  It is very slow,
so only use tiny inputs there.
"""

from __future__ import annotations

import math

import numpy as np
from numba import config, cuda

from .cpu_baseline import INVALID, RawHit, check_max_mm, hits_from_counts
from .encoding import C, G, GUIDE_LEN, PAM_LEN, SITE_LEN, reverse_complement

THREADS_PER_BLOCK = 256


def using_simulator() -> bool:
    return bool(config.ENABLE_CUDASIM)


def require_gpu() -> None:
    """Fail early with a clear message instead of a confusing CUDA error."""
    if not cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU found. On a Mac, set NUMBA_ENABLE_CUDASIM=1 to run the "
            "GPU code on the (slow) simulator with tiny inputs, or use "
            "--backend numba-cpu."
        )


@cuda.jit
def mismatch_kernel(seq, n_valid, guide, rc_guide, out_plus, out_minus):
    """Each thread i checks the 23-base window starting at seq[i]."""
    i = cuda.grid(1)  # this thread's global index = window start position
    if i < n_valid:
        # + strand: seq[i..i+19] vs guide, PAM seq[i+21], seq[i+22] == G G
        if seq[i + GUIDE_LEN + 1] == G and seq[i + GUIDE_LEN + 2] == G:
            mm = 0
            for k in range(GUIDE_LEN):
                if seq[i + k] != guide[k]:
                    mm += 1
            out_plus[i] = mm
        else:
            out_plus[i] = INVALID

        # - strand: seq[i], seq[i+1] == C C, then seq[i+3..i+22] vs rc(guide)
        if seq[i] == C and seq[i + 1] == C:
            mm = 0
            for k in range(GUIDE_LEN):
                if seq[i + PAM_LEN + k] != rc_guide[k]:
                    mm += 1
            out_minus[i] = mm
        else:
            out_minus[i] = INVALID
        # Every thread always writes its slot (count or INVALID).  That matters
        # because the output buffers are reused: no stale value from a previous
        # chunk can survive in positions 0 .. n_valid-1.


class GpuSearcher:
    """Holds device buffers for one guide and reuses them across chunks."""

    def __init__(self, guide: np.ndarray):
        require_gpu()
        guide = np.ascontiguousarray(guide, dtype=np.uint8)
        self.guide = guide
        self.d_guide = cuda.to_device(guide)
        self.d_rc_guide = cuda.to_device(np.ascontiguousarray(reverse_complement(guide)))
        self.capacity = 0  # largest chunk the buffers can hold
        self.allocations = 0  # how many times we had to (re)allocate; for tests
        self.d_seq = self.d_plus = self.d_minus = None

    def _ensure_capacity(self, n: int) -> None:
        if n <= self.capacity:
            return
        n_windows = n - SITE_LEN + 1
        self.d_seq = cuda.device_array(n, dtype=np.uint8)
        self.d_plus = cuda.device_array(n_windows, dtype=np.uint8)
        self.d_minus = cuda.device_array(n_windows, dtype=np.uint8)
        self.capacity = n
        self.allocations += 1

    # The three steps of one chunk, kept separate so benchmarks can time the
    # kernel on its own as well as the whole round trip.
    def upload(self, seq: np.ndarray) -> int:
        """Copy a chunk host -> device.  Returns n_valid (number of windows)."""
        seq = np.ascontiguousarray(seq, dtype=np.uint8)
        n = len(seq)
        n_valid = n - SITE_LEN + 1
        if n_valid > 0:
            self._ensure_capacity(n)
            self.d_seq[:n].copy_to_device(seq)
        return max(n_valid, 0)

    def launch(self, n_valid: int) -> None:
        """Start the kernel.  Returns immediately; the GPU keeps working."""
        if n_valid <= 0:
            return
        blocks = math.ceil(n_valid / THREADS_PER_BLOCK)
        mismatch_kernel[blocks, THREADS_PER_BLOCK](
            self.d_seq, n_valid, self.d_guide, self.d_rc_guide, self.d_plus, self.d_minus
        )

    def download(self, n_valid: int) -> tuple[np.ndarray, np.ndarray]:
        """Copy the counts device -> host.

        copy_to_host waits for the kernel to finish first, so no explicit
        cuda.synchronize() is needed here.
        """
        if n_valid <= 0:
            empty = np.empty(0, dtype=np.uint8)
            return empty, empty.copy()
        return self.d_plus[:n_valid].copy_to_host(), self.d_minus[:n_valid].copy_to_host()

    def count(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-position mismatch counts (INVALID where there is no PAM)."""
        n_valid = self.upload(seq)
        self.launch(n_valid)
        return self.download(n_valid)

    def search(self, seq: np.ndarray, max_mm: int) -> list[RawHit]:
        check_max_mm(max_mm)
        return hits_from_counts(*self.count(seq), max_mm)


# One searcher per guide, so repeated calls (one per chunk) reuse buffers.
_searchers: dict[bytes, GpuSearcher] = {}


def search_chunk_gpu(seq: np.ndarray, guide: np.ndarray, max_mm: int) -> list[RawHit]:
    """Backend function with the same signature as the CPU backends."""
    key = np.asarray(guide, dtype=np.uint8).tobytes()
    searcher = _searchers.get(key)
    if searcher is None:
        searcher = _searchers[key] = GpuSearcher(guide)
    return searcher.search(seq, max_mm)


def gpu_memory_info() -> tuple[int, int]:
    """(free, total) bytes on the current GPU."""
    return cuda.current_context().get_memory_info()
