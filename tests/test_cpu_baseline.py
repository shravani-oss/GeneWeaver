import random
from pathlib import Path

import numpy as np
import pytest

from geneweaver.cpu_baseline import search_chunk_numba, search_chunk_python
from geneweaver.encoding import encode, reverse_complement_str, validate_guide
from geneweaver.parser import iter_chunks
from geneweaver.pipeline import scan
from geneweaver.results import Hit, dedupe_hits, make_hit, write_tsv

MOCK = Path(__file__).parent / "data" / "mock.fa"
GUIDE = "GAGTCCGAGCAGAAGAAGAA"
BACKENDS = ["cpu", "numba-cpu"]

# Written by hand from the planting plan in tests/data/make_mock.py.
# (chrom, start, strand, mismatches, mismatch_positions, site)
EXPECTED = [
    ("chrA", 20, "+", 0, (), "GAGTCCGAGCAGAAGAAGAATGG"),
    ("chrA", 80, "+", 1, (20,), "GAGTCCGAGCAGAAGAAGAcAGG"),
    ("chrB", 90, "+", 1, (5,), "GAGTgCGAGCAGAAGAAGAATGG"),
    ("chrB", 277, "-", 1, (11,), "GAGTCCGAGCcGAAGAAGAAGGG"),
    ("chrA", 140, "-", 2, (1, 12), "tAGTCCGAGCAtAAGAAGAACGG"),
    ("chrB", 150, "+", 2, (7, 15), "GAGTCCnAGCAGAAtAAGAAAGG"),
    ("chrA", 200, "+", 3, (3, 10, 17), "GAtTCCGAGgAGAAGAcGAAGGG"),
    ("chrA", 260, "-", 4, (2, 8, 14, 19), "GcGTCCGcGCAGAcGAAGcATGG"),
]
FIVE_MM = ("chrA", 320, "+", 5, (1, 5, 9, 13, 17), "tAGTgCGAtCAGcAGAcGAAAGG")


def as_tuples(hits):
    return [(h.chrom, h.start, h.strand, h.mismatches, h.mismatch_positions, h.site) for h in hits]


# ---------------------------------------------------------------------------
# Planted sites in mock.fa
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("chunk_size", [23, 50, 100, 101, 1000, 1_000_000])
def test_mock_planted_sites(backend, chunk_size):
    hits = scan(MOCK, GUIDE, max_mm=4, backend=backend, chunk_size=chunk_size)
    assert as_tuples(hits) == EXPECTED


def test_border_site_really_crosses_a_chunk_border():
    """With chunk_size=100, chrB:90-112 must not fit in chrB's first chunk."""
    first_b = next(c for c in iter_chunks(MOCK, chunk_size=100) if c.chrom == "chrB")
    assert first_b.offset == 0 and first_b.end == 100
    assert 90 + 23 > first_b.end


@pytest.mark.parametrize("backend", BACKENDS)
def test_five_mismatches_only_with_max_mm_5(backend):
    hits = as_tuples(scan(MOCK, GUIDE, max_mm=5, backend=backend, chunk_size=100))
    assert hits == EXPECTED + [FIVE_MM]


@pytest.mark.parametrize("backend", BACKENDS)
def test_max_mm_0_is_on_target_only(backend):
    hits = scan(MOCK, GUIDE, max_mm=0, backend=backend)
    assert as_tuples(hits) == EXPECTED[:1]
    assert hits[0].label == "on-target"


def test_labels():
    hits = scan(MOCK, GUIDE, max_mm=4)
    assert [h.label for h in hits] == ["on-target"] + ["off-target"] * 7


# ---------------------------------------------------------------------------
# Python and Numba agree with an independent, string-based reference
# ---------------------------------------------------------------------------
def reference_search(seq: str, guide: str, max_mm: int):
    """Deliberately different, very literal implementation (strings, no codes)."""
    seq = "".join(b if b in "ACGT" else "N" for b in seq.upper())
    rc_target = reverse_complement_str(guide + "NGG")  # "CCN" + rc(guide)
    hits = []
    for p in range(len(seq) - 22):
        w = seq[p : p + 23]
        if w[21:23] == "GG":
            mm = sum(a != b for a, b in zip(w[:20], guide))
            if mm <= max_mm:
                hits.append((p, "+", mm))
        if w[0:2] == "CC":
            mm = sum(a != b for a, b in zip(w[3:], rc_target[3:]))
            if mm <= max_mm:
                hits.append((p, "-", mm))
    return hits


def hit_rich_sequence(rng: random.Random, n_sites: int) -> str:
    """Random DNA stuffed with mutated guide copies on both strands, plus Ns."""
    parts = []
    for _ in range(n_sites):
        g = list(GUIDE)
        for _ in range(rng.randint(0, 6)):
            g[rng.randrange(20)] = rng.choice("ACGTN")
        site = "".join(g) + rng.choice("ACGTN") + rng.choice("GGGN") + rng.choice("GGGN")
        if rng.random() < 0.5:
            site = reverse_complement_str(site)
        parts.append("".join(rng.choice("ACGTN") for _ in range(rng.randint(0, 15))))
        parts.append(site)
    return "".join(parts)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("max_mm", [0, 2, 4, 6])
def test_backends_match_reference(seed, max_mm):
    rng = random.Random(seed)
    seq = hit_rich_sequence(rng, 300)
    guide = validate_guide(GUIDE)
    expected = reference_search(seq, GUIDE, max_mm)
    assert len(expected) > 0
    enc = encode(seq)
    assert search_chunk_python(enc, guide, max_mm) == expected
    assert search_chunk_numba(enc, guide, max_mm) == expected


@pytest.mark.parametrize("search", [search_chunk_python, search_chunk_numba])
@pytest.mark.parametrize("length", [0, 1, 22, 23])
def test_tiny_inputs(search, length):
    seq = encode((GUIDE + "AGG")[:length])
    hits = search(seq, validate_guide(GUIDE), 4)
    assert hits == ([(0, "+", 0)] if length == 23 else [])


def test_max_mm_out_of_range():
    with pytest.raises(ValueError):
        search_chunk_python(encode("A" * 30), validate_guide(GUIDE), 21)


# ---------------------------------------------------------------------------
# results.py
# ---------------------------------------------------------------------------
def test_dedupe_keeps_one_per_key():
    h = Hit("chr1", 5, "+", "A" * 23, 1, (3,))
    other_strand = Hit("chr1", 5, "-", "A" * 23, 1, (3,))
    assert dedupe_hits([h, h, other_strand, h]) == [h, other_strand]


def test_make_hit_catches_wrong_mismatch_count():
    seq = encode(GUIDE + "TGG")
    with pytest.raises(AssertionError):
        make_hit("c", 0, seq, 0, "+", 2, validate_guide(GUIDE))


def test_write_tsv(tmp_path):
    out = tmp_path / "sub" / "hits.tsv"
    write_tsv(scan(MOCK, GUIDE, max_mm=4), out)
    lines = out.read_text().splitlines()
    assert lines[0].split("\t") == [
        "chrom", "start", "strand", "site", "mismatches", "mismatch_positions", "score", "label",
    ]
    assert lines[1].split("\t") == ["chrA", "20", "+", "GAGTCCGAGCAGAAGAAGAATGG", "0", ".", ".", "on-target"]
    assert lines[5].split("\t")[5] == "1,12"
    assert len(lines) == 1 + len(EXPECTED)
