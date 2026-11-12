"""Generate tests/data/mock.fa: a tiny genome with hand-planted CRISPR sites.

Run from the repo root:  python tests/data/make_mock.py

Background is random (fixed seed, so the file is always identical).  The
expected hits are written out by hand in tests/test_cpu_baseline.py; if you
change anything here, update them there too.

Guide: GAGTCCGAGCAGAAGAAGAA (EMX1 guide, Hsu et al. 2013)
"""

from __future__ import annotations

import random
from pathlib import Path

GUIDE = "GAGTCCGAGCAGAAGAAGAA"
# Deterministic substitution: always a different base.
SUB = {"A": "C", "C": "G", "G": "T", "T": "A"}
COMP = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def rc(s: str) -> str:
    return "".join(COMP[b] for b in reversed(s))


def target(mm_positions=(), pam="TGG", n_positions=()):
    """guide with substitutions at 1-based positions, N at n_positions, then PAM."""
    g = list(GUIDE)
    for p in mm_positions:
        g[p - 1] = SUB[g[p - 1]]
    for p in n_positions:
        g[p - 1] = "N"
    return "".join(g) + pam


def plant(seq: list[str], start: int, strand: str, site: str, lowercase=False):
    """Write a 23-base site into the forward strand at `start`."""
    assert len(site) == 23
    forward = site if strand == "+" else rc(site)
    if lowercase:
        forward = forward.lower()
    seq[start : start + 23] = list(forward)


def background(rng: random.Random, n: int) -> list[str]:
    return [rng.choice("ACGT") for _ in range(n)]


def build() -> list[tuple[str, str]]:
    rng = random.Random(2026)

    # ---- chrA: one site for each mismatch count 0..5 ----
    a = background(rng, 400)
    plant(a, 20, "+", target(pam="TGG"))                                 # 0 mm (on-target)
    plant(a, 80, "+", target([20], pam="AGG"))                           # 1 mm next to PAM
    plant(a, 140, "-", target([1, 12], pam="CGG"))                       # 2 mm, minus strand
    plant(a, 200, "+", target([3, 10, 17], pam="GGG"), lowercase=True)   # 3 mm, soft-masked
    plant(a, 260, "-", target([2, 8, 14, 19], pam="TGG"))                # 4 mm, minus strand
    plant(a, 320, "+", target([1, 5, 9, 13, 17], pam="AGG"))             # 5 mm -> NOT reported

    # ---- chrB: edge cases ----
    b = background(rng, 300)
    b[30:40] = list("NNNNNNNNNN")                                        # run of unknown bases
    b[45] = "R"                                                          # IUPAC code -> N
    plant(b, 90, "+", target([5], pam="TGG"))                            # crosses 100 (chunk_size=100 border)
    plant(b, 150, "+", target([15], pam="AGG", n_positions=[7]))         # N in guide -> 2 mm
    plant(b, 210, "+", target(pam="TGN"))                                # N in PAM G -> invalid
    plant(b, 277, "-", target([11], pam="GGG"))                          # last 23 bases of chrB

    # ---- chrC: shorter than one site, must not crash ----
    c = list("ACGTACGTAC")

    return [("chrA mock chromosome A", "".join(a)), ("chrB", "".join(b)), ("chrC", "".join(c))]


def main() -> None:
    out = Path(__file__).with_name("mock.fa")
    with open(out, "w") as fh:
        for name, seq in build():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i : i + 60] + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
