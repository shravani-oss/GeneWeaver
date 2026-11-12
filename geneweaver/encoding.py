"""Turn DNA letters into small numbers, and back.

GPUs are much faster at comparing numbers than text, so every base is stored
as a single byte (``uint8``):

    A=0, C=1, G=2, T=3, N=4

Anything that is not A/C/G/T (for example ``N``, or IUPAC codes like ``R``)
becomes ``N``. Lowercase letters (UCSC uses them for repeat regions) are
treated exactly like uppercase ones.
"""

from __future__ import annotations

import numpy as np

# ----- base codes -----------------------------------------------------------
A, C, G, T, N = 0, 1, 2, 3, 4

# ----- SpCas9 target-site layout --------------------------------------------
# A site is 20 guide bases followed by a 3-base PAM ("NGG"): 23 bases total.
GUIDE_LEN = 20
PAM_LEN = 3
SITE_LEN = GUIDE_LEN + PAM_LEN

# Lookup table: index = ASCII code of a letter, value = its number.
# Using a table means a whole line of text is encoded in one NumPy step,
# instead of a slow Python loop over every letter.
_ENCODE_LUT = np.full(256, N, dtype=np.uint8)
for _code, _letter in enumerate("ACGT"):
    _ENCODE_LUT[ord(_letter)] = _code
    _ENCODE_LUT[ord(_letter.lower())] = _code

# Reverse table: number -> ASCII letter.
_DECODE_LUT = np.frombuffer(b"ACGTN", dtype=np.uint8)

# Complement of each code: A<->T, C<->G, N stays N.
_COMPLEMENT = np.array([T, G, C, A, N], dtype=np.uint8)


def encode_bytes(buf: bytes) -> np.ndarray:
    """Encode raw ASCII bytes (e.g. one FASTA line) into a uint8 array."""
    return _ENCODE_LUT[np.frombuffer(buf, dtype=np.uint8)]


def encode(seq: str) -> np.ndarray:
    """Encode a DNA string like ``"ACGTn"`` into ``[0, 1, 2, 3, 4]``."""
    return encode_bytes(seq.encode("ascii"))


def decode(codes: np.ndarray) -> str:
    """Turn a uint8 code array back into an uppercase DNA string."""
    return _DECODE_LUT[np.asarray(codes, dtype=np.uint8)].tobytes().decode("ascii")


def reverse_complement(codes: np.ndarray) -> np.ndarray:
    """Reverse complement of an encoded sequence.

    This is the other DNA strand read in its own 5'->3' direction:
    read backwards, swapping A<->T and C<->G.
    """
    return _COMPLEMENT[np.asarray(codes, dtype=np.uint8)[::-1]]


def reverse_complement_str(seq: str) -> str:
    """Reverse complement of a DNA string (non-ACGT letters become N)."""
    return decode(reverse_complement(encode(seq)))


def validate_guide(guide: str) -> np.ndarray:
    """Check a guide is exactly 20 letters of A/C/G/T and return it encoded."""
    cleaned = guide.strip().upper()
    if len(cleaned) != GUIDE_LEN:
        raise ValueError(
            f"guide must be {GUIDE_LEN} bases long, got {len(cleaned)}: {guide!r}"
        )
    bad = sorted(set(cleaned) - set("ACGT"))
    if bad:
        raise ValueError(f"guide may only contain A/C/G/T, found {bad} in {guide!r}")
    return encode(cleaned)
