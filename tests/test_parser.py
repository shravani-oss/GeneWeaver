import gzip
import random

import pytest

from geneweaver.encoding import SITE_LEN, decode
from geneweaver.parser import CHUNK_OVERLAP, iter_chunks


def write_fasta(path, records, width=60):
    """records: list of (header, sequence)."""
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i : i + width] + "\n")


def rebuild(chunks):
    """Glue chunks back together by dropping each chunk's overlap."""
    out = {}
    for ch in chunks:
        s = decode(ch.seq)
        if ch.chrom not in out:
            assert ch.offset == 0
            out[ch.chrom] = s
        else:
            assert ch.offset == len(out[ch.chrom]) - CHUNK_OVERLAP
            out[ch.chrom] += s[CHUNK_OVERLAP:]
    return out


@pytest.fixture
def genome(tmp_path):
    rng = random.Random(1)
    recs = [
        ("chr1 some description", "".join(rng.choice("ACGTacgtN") for _ in range(1000))),
        ("chr2", "".join(rng.choice("ACGT") for _ in range(257))),
        ("tiny", "ACGTACGT"),
    ]
    path = tmp_path / "g.fa"
    write_fasta(path, recs, width=61)
    return path, recs


@pytest.mark.parametrize("chunk_size", [23, 24, 50, 100, 1000, 10_000])
def test_chunks_rebuild_original(genome, chunk_size):
    path, recs = genome
    chunks = list(iter_chunks(path, chunk_size=chunk_size))
    assert all(len(c.seq) <= chunk_size for c in chunks)
    rebuilt = rebuild(chunks)
    assert rebuilt == {
        "chr1": recs[0][1].upper(),
        "chr2": recs[1][1],
        "tiny": "ACGTACGT",
    }


@pytest.mark.parametrize("chunk_size", [23, 37, 100, 999])
def test_every_window_in_exactly_one_chunk(genome, chunk_size):
    """Each 23-base window start must be searchable in exactly one chunk."""
    path, recs = genome
    starts = {}
    for ch in iter_chunks(path, chunk_size=chunk_size):
        for i in range(len(ch.seq) - SITE_LEN + 1):
            starts.setdefault(ch.chrom, []).append(ch.offset + i)
    for header, seq in recs:
        name = header.split()[0]
        expected = list(range(len(seq) - SITE_LEN + 1))
        assert sorted(starts.get(name, [])) == expected


def test_header_uses_first_word_and_lowercase_is_uppercased(genome):
    path, _ = genome
    chunks = list(iter_chunks(path))
    assert [c.chrom for c in chunks] == ["chr1", "chr2", "tiny"]
    assert "a" not in decode(chunks[0].seq)


def test_gzip(tmp_path, genome):
    path, recs = genome
    gz = tmp_path / "g.fa.gz"
    with open(path, "rb") as src, gzip.open(gz, "wb") as dst:
        dst.write(src.read())
    assert rebuild(iter_chunks(gz, chunk_size=100)) == rebuild(iter_chunks(path, chunk_size=100))


def test_main_chroms_only(tmp_path):
    path = tmp_path / "hg.fa"
    write_fasta(path, [
        ("chr1", "A" * 30),
        ("chr1_KI270706v1_random", "C" * 30),
        ("chrUn_GL000195v1", "G" * 30),
        ("chrX", "T" * 30),
    ])
    assert [c.chrom for c in iter_chunks(path)] == ["chr1", "chr1_KI270706v1_random", "chrUn_GL000195v1", "chrX"]
    assert [c.chrom for c in iter_chunks(path, main_chroms_only=True)] == ["chr1", "chrX"]


def test_blank_lines_and_crlf(tmp_path):
    path = tmp_path / "crlf.fa"
    path.write_bytes(b">c\r\nACGT\r\n\r\nNNac\r\n")
    (ch,) = iter_chunks(path)
    assert decode(ch.seq) == "ACGTNNAC"


def test_chunk_size_too_small(genome):
    path, _ = genome
    with pytest.raises(ValueError):
        list(iter_chunks(path, chunk_size=CHUNK_OVERLAP))


def test_sequence_before_header(tmp_path):
    path = tmp_path / "bad.fa"
    path.write_text("ACGT\n>c\nACGT\n")
    with pytest.raises(ValueError):
        list(iter_chunks(path))


@pytest.mark.parametrize("chunk_size", [23, 24, 37, 100, 257, 999, 1000, 10_000])
def test_count_chunks_matches_iter_chunks(genome, chunk_size):
    from geneweaver.parser import count_chunks

    path, _ = genome
    assert count_chunks(path, chunk_size) == len(list(iter_chunks(path, chunk_size)))


def test_record_lengths_uses_fai_and_filters(tmp_path):
    from geneweaver.parser import record_lengths

    path = tmp_path / "hg.fa"
    write_fasta(path, [("chr1", "A" * 30), ("chrUn_x", "C" * 7)])
    assert record_lengths(path) == {"chr1": 30, "chrUn_x": 7}
    # A (deliberately different) .fai is preferred when present
    (tmp_path / "hg.fa.fai").write_text("chr1\t99\t6\t60\t61\nchrUn_x\t5\t9\t60\t61\n")
    assert record_lengths(path) == {"chr1": 99, "chrUn_x": 5}
    assert record_lengths(path, main_chroms_only=True) == {"chr1": 99}
