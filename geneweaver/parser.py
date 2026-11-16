"""Read a FASTA genome piece by piece and cut it into overlapping chunks.

Why chunks?  A whole human genome is ~3.1 billion bases.  We never load it all
at once (it would not fit on the Mac, and a GPU only has 16 GB).  Instead we
read the file line by line and hand out fixed-size pieces ("chunks").

Why overlap?  A target site is 23 bases long.  If we cut the genome at an
arbitrary point, a site could be split in half and missed.  So each chunk
starts 22 bases (site length - 1) before the end of the previous chunk:

    chunk 0: [0 ............ 100)
    chunk 1:          [78 ............ 178)       (with chunk_size=100)

Every 23-base window now fits completely inside at least one chunk.

Why not Bio.SeqIO.parse?  SeqIO builds each chromosome as one big string
(chr1 is ~250 MB), so memory would grow with the chromosome size.  Reading
lines ourselves keeps memory at roughly one chunk, no matter how big the
genome is.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

from .encoding import SITE_LEN, encode_bytes

CHUNK_OVERLAP = SITE_LEN - 1  # 22
DEFAULT_CHUNK_SIZE = 50_000_000

# hg38 also contains "_alt", "_random" and "chrUn" contigs; these are the
# primary chromosomes kept by --main-chroms-only.
MAIN_CHROMS = frozenset([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])


@dataclass(frozen=True)
class Chunk:
    """One piece of one chromosome.

    ``seq[i]`` is the base at 0-based genome position ``offset + i``.
    """

    chrom: str
    offset: int
    seq: np.ndarray  # uint8 codes, see encoding.py

    @property
    def end(self) -> int:
        return self.offset + len(self.seq)


def _open(path: str | Path) -> BinaryIO:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb")


def iter_chunks(
    path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    main_chroms_only: bool = False,
) -> Iterator[Chunk]:
    """Stream a FASTA (optionally .gz) file as overlapping encoded chunks.

    Each yielded chunk is at most ``chunk_size`` bases.  Consecutive chunks of
    the same chromosome share ``CHUNK_OVERLAP`` (22) bases.  Chunks never span
    two chromosomes.
    """
    if chunk_size <= CHUNK_OVERLAP:
        raise ValueError(f"chunk_size must be > {CHUNK_OVERLAP}, got {chunk_size}")

    # One reusable buffer.  We fill it line by line; when it is full we hand
    # out a copy and slide the last 22 bases to the front.
    buf = np.empty(chunk_size, dtype=np.uint8)
    fill = 0  # how many bases are in buf right now
    fresh = 0  # how many of those have not been handed out yet
    offset = 0  # genome position of buf[0]
    chrom: str | None = None
    keep = False

    def flush() -> Chunk:
        return Chunk(chrom, offset, buf[:fill].copy())

    with _open(path) as fh:
        for line in fh:
            if line.startswith(b">"):
                # New record: finish the previous one first.
                if keep and fresh > 0:
                    yield flush()
                header = line[1:].strip()
                chrom = header.split()[0].decode("ascii") if header else ""
                keep = (not main_chroms_only) or chrom in MAIN_CHROMS
                fill = fresh = offset = 0
                continue

            if chrom is None:
                if line.strip():
                    raise ValueError(f"{path}: sequence data before the first '>' header")
                continue
            if not keep:
                continue

            codes = encode_bytes(line.rstrip())
            pos = 0
            while pos < len(codes):
                take = min(chunk_size - fill, len(codes) - pos)
                buf[fill : fill + take] = codes[pos : pos + take]
                fill += take
                fresh += take
                pos += take
                if fill == chunk_size:
                    yield flush()
                    # Keep the last 22 bases as the start of the next chunk.
                    buf[:CHUNK_OVERLAP] = buf[chunk_size - CHUNK_OVERLAP :]
                    offset += chunk_size - CHUNK_OVERLAP
                    fill = CHUNK_OVERLAP
                    fresh = 0

    if keep and fresh > 0:
        yield flush()


# ---------------------------------------------------------------------------
# Counting chunks ahead of time (for progress bars)
# ---------------------------------------------------------------------------
def chunks_for_length(length: int, chunk_size: int) -> int:
    """How many chunks ``iter_chunks`` yields for a chromosome of ``length`` bases."""
    if length <= 0:
        return 0
    if length <= chunk_size:
        return 1
    step = chunk_size - CHUNK_OVERLAP
    return 1 + -(-(length - chunk_size) // step)  # ceiling division


def record_lengths(path: str | Path, main_chroms_only: bool = False) -> dict[str, int]:
    """Length of every record, without keeping any sequence in memory.

    Uses the ``.fai`` index (made by ``samtools faidx``) if one sits next to
    the file, which is instant.  Otherwise reads the file once, counting bases.
    """
    path = Path(path)
    fai = Path(str(path) + ".fai")
    lengths: dict[str, int] = {}
    if fai.exists() and path.suffix != ".gz":
        for line in fai.read_text().splitlines():
            name, length = line.split("\t")[:2]
            lengths[name] = int(length)
    else:
        name = None
        with _open(path) as fh:
            for line in fh:
                if line.startswith(b">"):
                    header = line[1:].strip()
                    name = header.split()[0].decode("ascii") if header else ""
                    lengths[name] = 0
                elif name is not None:
                    lengths[name] += len(line.rstrip())
    if main_chroms_only:
        lengths = {k: v for k, v in lengths.items() if k in MAIN_CHROMS}
    return lengths


def count_chunks(path: str | Path, chunk_size: int = DEFAULT_CHUNK_SIZE, main_chroms_only: bool = False) -> int:
    return sum(chunks_for_length(n, chunk_size) for n in record_lengths(path, main_chroms_only).values())
