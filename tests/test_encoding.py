import numpy as np
import pytest

from geneweaver.encoding import (
    A, C, G, N, T,
    decode,
    encode,
    reverse_complement,
    reverse_complement_str,
    validate_guide,
)


def test_encode_basic():
    assert encode("ACGTN").tolist() == [A, C, G, T, N]
    assert encode("ACGTN").dtype == np.uint8


def test_lowercase_is_same_as_uppercase():
    assert encode("acgtn").tolist() == encode("ACGTN").tolist()


def test_other_letters_become_n():
    # IUPAC ambiguity codes and junk all map to N
    assert encode("RYKMSWBDHV-*").tolist() == [N] * 12


def test_decode_roundtrip():
    seq = "GATTACANNCG"
    assert decode(encode(seq)) == seq
    assert decode(encode(seq.lower())) == seq  # decode is always uppercase


def test_reverse_complement():
    assert reverse_complement_str("AACG") == "CGTT"
    assert reverse_complement_str("ACGTN") == "NACGT"
    assert reverse_complement(encode("GGG")).tolist() == [C, C, C]


def test_reverse_complement_twice_is_identity():
    seq = "GAGTCCGAGCAGAAGAAGAANGG"
    assert reverse_complement_str(reverse_complement_str(seq)) == seq


def test_validate_guide_ok():
    g = validate_guide("gagtccgagcagaagaagaa")
    assert decode(g) == "GAGTCCGAGCAGAAGAAGAA"


@pytest.mark.parametrize("bad", ["ACGT", "A" * 21, "GAGTCCGAGCAGAAGAAGAN"])
def test_validate_guide_rejects_bad(bad):
    with pytest.raises(ValueError):
        validate_guide(bad)
