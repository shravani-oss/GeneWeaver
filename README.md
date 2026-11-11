# GeneWeaver: GPU-Accelerated CRISPR Off-Target Search Engine

GeneWeaver scans a genome (a `.fasta` file) for places where a CRISPR guide could accidentally cut ("off-target sites"). It uses custom NVIDIA GPU kernels written in Python with Numba, spreads the work over several GPUs with Dask, and shows live progress in a terminal dashboard built with Textual.

---

## 0. Read this first (instructions for Claude Code)

- **Developer's machine:** MacBook Air, Apple M1, 8 GB RAM. It has **no NVIDIA GPU**, so CUDA code cannot run on it.
- **Never** install the CUDA Toolkit or NVIDIA drivers on the Mac.
- Real GPU runs happen on **Kaggle** (accelerator "GPU T4 x2" = 2 × NVIDIA T4, 16 GB each). The final live terminal demo runs on **Lightning AI** (or AWS), because a notebook can't display a TUI.
- On the Mac, test GPU kernels with the **CUDA simulator** (`NUMBA_ENABLE_CUDASIM=1`), using tiny inputs only, because it is very slow.
- **Never load the full genome into memory on the Mac.** Always stream and chunk.
- **Correctness rule:** GPU results must **exactly match** CPU baseline results on the same input. A speedup only counts if this holds.
- Work **one stage at a time**. Finish every "Done when" check, then stop and report before starting the next stage.
- Commit to Git after each working step. Never commit genome data.
- Keep code compatible with **Python 3.10+**, because Kaggle's Python version may differ from the local 3.12.
- **Do not invent scientific constants** (for example, scoring weights). Take them from the published source and cite it in a code comment.
- The developer is learning CUDA, so explain every change in simple language.

---

## 1. Which machine does what

| Machine | Role |
|---|---|
| **MacBook (VS Code)** | Write code, CPU baseline, unit tests, CUDA simulator tests, Dask on CPU, TUI layout |
| **GitHub** | The bridge: push from the Mac, then `git clone` / `git pull` on the GPU machine |
| **Kaggle Notebook (GPU T4 x2)** | Real CUDA runs, benchmarks, full-genome memory test, 2-GPU Dask |
| **Lightning AI / AWS (GPU + terminal)** | Final live TUI demo with real GPU stats |

The **whole pipeline runs on one machine at a time**. The Mac never sends work to a cloud GPU while the program is running.

```
FASTA file
   │  parser.py: stream → uppercase/clean → encode to numbers → chunk with overlap
   ▼
Chunks (NumPy uint8 arrays)
   │  scheduler.py: Dask, one worker per GPU
   ▼
gpu_kernels.py: one GPU thread checks one genome position
   │
   ▼
Raw hits → scoring.py (MIT / CFD) → ranked results (TSV)
   │
   ▼
tui.py (Textual): progress bars, GPU memory/utilization, hits with mismatches in red
```

---

## 2. The search problem (specification)

- **Inputs:** a FASTA genome, a guide sequence (20 letters of A/C/G/T), and the maximum number of mismatches (default `4`).
- **Target site (SpCas9):** 20 bases that match the guide, followed directly by a 3-base **PAM** `NGG`, for 23 bases in total.
  - In the PAM, `N` means any base, and both `G`s must match exactly.
  - Mismatches are counted only in the 20 guide bases.
- **Both DNA strands must be searched.** Don't build a reverse-complement copy of the genome. Instead, also search the forward strand for the pattern `CCN` + reverse_complement(guide), and report those hits with strand `-`.
- **Unknown bases (`N`) in the genome:** an `N` in the guide region counts as a mismatch. A window with `N` in a required PAM `G` (or `C` for the minus strand) is invalid.
- **Encoding:** `A=0, C=1, G=2, T=3, N=4`, stored as `uint8`. Uppercase everything first, because UCSC files use lowercase letters for repeat regions.
- **Chunk overlap:** each chunk starts 22 bases (site length − 1) before the end of the previous chunk, so no site that crosses a chunk border is missed. Remove duplicate hits by `(chrom, start, strand)`.
- **Coordinates:** 0-based genome position, computed as chunk offset + local index.
- **Output (TSV):** `chrom, start, strand, site, mismatches, mismatch_positions, score`. In `site`, mismatched bases are written in lowercase.
- **On-target site:** a site with 0 mismatches is labeled `on-target`, not counted as an off-target.

---

## 3. Project structure

```
GeneWeaver/
├── README.md
├── requirements.txt          # Mac / CPU packages
├── requirements-gpu.txt      # extra packages for NVIDIA machines only
├── pyproject.toml
├── .gitignore                # data/, results/, .venv/, *.fa, *.fa.gz
├── geneweaver/
│   ├── __init__.py
│   ├── __main__.py           # CLI: python -m geneweaver scan ...
│   ├── encoding.py           # letters <-> numbers, reverse complement
│   ├── parser.py             # FASTA streaming, cleaning, chunking
│   ├── cpu_baseline.py       # pure-Python search (+ optional Numba CPU @njit)
│   ├── gpu_kernels.py        # Numba CUDA kernels
│   ├── scheduler.py          # Dask: CPU LocalCluster or multi-GPU LocalCUDACluster
│   ├── scoring.py            # MIT and CFD scores
│   ├── results.py            # hit records, dedupe, TSV writer
│   └── tui.py                # Textual dashboard
├── tests/
│   ├── data/mock.fa          # tiny hand-made genome with planted sites
│   ├── test_encoding.py
│   ├── test_parser.py
│   ├── test_cpu_baseline.py
│   ├── test_gpu_matches_cpu.py   # runs on the simulator locally, on a real GPU on Kaggle
│   └── test_scoring.py
├── benchmarks/
│   └── bench.py              # CPU vs GPU timing, writes benchmarks/results.md
├── notebooks/
│   └── kaggle_runner.ipynb   # clone → install → test → run → benchmark
└── data/                     # genome files (gitignored)
```

**CLI target:**
```
python -m geneweaver scan --fasta data/chr21.fa --guide GAGTCCGAGCAGAAGAAGAA \
    --max-mm 4 --backend {cpu,numba-cpu,gpu} --gpus 2 --chunk-size 50000000 \
    --out results/hits.tsv --tui
```

---

## 4. Setup

### 4.1 Mac (local development)

```bash
# Check tools
python3 --version
git --version
uname -m            # must print: arm64

# Project + virtual environment (Python 3.12)
mkdir -p ~/Projects/GeneWeaver && cd ~/Projects/GeneWeaver
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code, install the **Python**, **Pylance** and **Jupyter** extensions, then pick the interpreter: `Cmd+Shift+P` → *Python: Select Interpreter* → `.venv/bin/python`.

**requirements.txt**
```
numpy
biopython
numba
dask
distributed
textual
pytest
```
If `from numba import cuda` fails on the Mac, run `pip install numba-cuda` (without the `[cu12]` extra) so the simulator is available.

**requirements-gpu.txt** (install on NVIDIA machines only)
```
numba-cuda[cu12]
dask-cuda
nvidia-ml-py
```
`numba-cuda` replaces Numba's deprecated built-in CUDA target; your code still uses `from numba import cuda`. `nvidia-ml-py` provides the `pynvml` module, which the TUI uses for GPU stats.

**Run kernel tests on the Mac (simulator):**
```bash
NUMBA_ENABLE_CUDASIM=1 pytest tests/test_gpu_matches_cpu.py -q
```

### 4.2 GitHub
Create a repo named `GeneWeaver` and push from the Mac. If the repo is private, put a GitHub token in **Kaggle → Add-ons → Secrets** and clone with it.

### 4.3 Kaggle (GPU runs)
1. Verify your phone number in your Kaggle account settings (needed for GPU and Internet).
2. Create a notebook. In **Settings**, set **Accelerator: GPU T4 x2** and turn **Internet: On**.
3. First cells:
```python
!nvidia-smi                       # should list 2 × Tesla T4
!git clone https://github.com/<your-username>/GeneWeaver.git
%cd GeneWeaver
!pip install -q -r requirements.txt -r requirements-gpu.txt
!pip install -q -e .
!python -m pytest tests -q
```
4. After every push from the Mac, run `!git pull` in the notebook.
5. GPU hours are limited each week. Turn the GPU off when you're only editing.

### 4.4 Data
Use datasets in this order: **mock file → one chromosome → full genome**.

```bash
mkdir -p data && cd data
# One chromosome (~46 million bases), fine on the Mac
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz
gunzip chr21.fa.gz
# Full human genome: download on Kaggle only, never on the Mac
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
```
`hg38.fa.gz` also has extra contigs (`_alt`, `_random`, `chrUn`). Add a `--main-chroms-only` flag that keeps chr1–22, X and Y.

---

## 5. Stages

### Stage 0: Environment (Mac + Kaggle)
- [ ] Mac checks pass, `.venv` created, `requirements.txt` installed
- [ ] Folder structure, `.gitignore`, `pyproject.toml` created; first commit pushed
- [ ] Kaggle account verified, notebook with GPU T4 x2 shows 2 GPUs in `nvidia-smi`
- [ ] `data/chr21.fa` downloaded on the Mac

**Done when:** `python -c "import Bio, numpy, numba, dask, textual"` works on the Mac, and `nvidia-smi` shows 2 GPUs on Kaggle.

### Week 1: Parser and CPU baseline (Mac)
- [ ] `encoding.py`: encode/decode, reverse complement, with tests
- [ ] `parser.py`: stream records with `Bio.SeqIO.parse`, uppercase, encode to `uint8`, yield chunks `(chrom, offset, array)` with a 22-base overlap. Memory must stay flat (no full-file load).
- [ ] `tests/data/mock.fa`: a small genome with **planted sites** (0, 1, 2, 3, 4 and 5 mismatches, both strands, one crossing a chunk border, one with `N`). The expected hits are written in the test.
- [ ] `cpu_baseline.py`: pure-Python loop over every position. Compare the 20 guide bases, check the PAM, and do this for both strands.
- [ ] Optional: a Numba CPU (`@njit`) version, for a fair "CPU at full speed" comparison
- [ ] `results.py`: dedupe across chunk overlaps, write TSV

**Done when:** all planted sites are found with the correct position, strand and mismatch count; the 5-mismatch site is **not** reported (max 4); and chr21 parses on the Mac without memory growing.

### Week 2: CUDA kernel and TUI layout
**Kernel (write on the Mac, run on Kaggle)**
- [ ] `gpu_kernels.py`: a `@cuda.jit` kernel where **one thread = one genome position**. Each thread writes its mismatch count into an output `uint8` array (`255` = invalid / no PAM). Use separate outputs for `+` and `-` strands.
- [ ] Launch config: `threads_per_block = 256`, `blocks = ceil(n / 256)`. Guard with `if i < n_valid`.
- [ ] Memory transfers: `cuda.to_device()` for the chunk and the guide, `cuda.device_array()` for outputs, `.copy_to_host()` for results. The host then finds hits with `np.nonzero(out <= max_mm)`.
- [ ] Reuse device buffers across chunks; don't allocate new ones for every chunk.
- [ ] `test_gpu_matches_cpu.py`: GPU hits == CPU hits on `mock.fa` (simulator on the Mac), then on chr21 (real GPU on Kaggle)

**TUI (Mac)**
- [ ] `tui.py`: a Textual app with a progress bar (chunks done / total), a results `DataTable`, and a GPU stats panel
- [ ] Run the scan in a background worker (`@work(thread=True)`) so the UI doesn't freeze
- [ ] GPU panel updates every second with `set_interval`. On the Mac, show "No NVIDIA GPU (CPU mode)" instead of crashing.

**Done when:** GPU results exactly equal CPU results on `mock.fa` and chr21, and the TUI shows live progress for a CPU-backend scan on the Mac.

### Mid-project review (Kaggle)
**Performance audit**
- [ ] `benchmarks/bench.py`: run the **CPU and GPU benchmarks on the same machine (Kaggle)** with the **same input**: a 5-million-base slice of chr21 (pure Python on all of chr21 would take hours)
- [ ] Report both kernel-only time and end-to-end time (including transfers), plus throughput (bases/second) for chr21 and the full genome on the GPU
- [ ] **Target: ≥ 100× faster than pure Python**

**Memory check**
- [ ] Scan the full `hg38` chunk by chunk (start with `--chunk-size 50000000`)
- [ ] Log free/total GPU memory after each chunk with `cuda.current_context().get_memory_info()`
- [ ] No `CUDA_ERROR_OUT_OF_MEMORY`, and GPU memory use stays flat across chunks (no leaks)

**Done when:** `benchmarks/results.md` shows ≥ 100× speedup and the full genome finishes without memory errors.

### Week 3: Multi-GPU and scoring
**Dask**
- [ ] `scheduler.py`: on GPU machines, `LocalCUDACluster()` creates one worker per GPU. Submit chunks with `client.map(...)` and gather the results.
- [ ] On the Mac, the same code runs with a CPU `LocalCluster` and the CPU backend, so you can test the scheduling logic locally
- [ ] Log which worker/GPU processed each chunk, how many bases, and how long it took

**Scoring**
- [ ] `scoring.py`: implement the **MIT score** (Hsu et al., 2013) and the **CFD score** (Doench et al., 2016) using the weights from the original publications (cite them in comments). Mismatches near the PAM (the "seed region") count as worse.
- [ ] Rank hits by score, from most to least dangerous; put the score in the TSV and the TUI
- [ ] Tests: a perfect match scores highest, and the same mismatch scores worse when it's closer to the PAM

**Done when:** on Kaggle T4 x2, each GPU processes 50% ± 10% of the bases, results equal the single-GPU results, and the 2-GPU run is faster than the 1-GPU run.

### Week 4: Optimization and polish
**Shared memory**
- [ ] A second kernel version: each block loads the guide plus a genome tile (`blockDim + 22` bases) into `cuda.shared.array`, calls `cuda.syncthreads()`, then threads read from the fast shared memory
- [ ] Benchmark it against the Week 2 kernel and report the result honestly, even if the gain is small
- [ ] Results must still exactly match the CPU baseline

**TUI polish**
- [ ] Live GPU memory and utilization per GPU via `pynvml`
- [ ] Hits table shows each site with mismatched bases in **red** (Rich markup `[red]a[/red]`)
- [ ] Summary panel: total hits, hits per mismatch count, run time, speedup

**Final demo (Lightning AI / AWS with a GPU and a terminal)**
- [ ] `git clone`, install both requirements files, run `python -m geneweaver scan ... --gpus 1 --tui`
- [ ] Record a short screen capture for the README

**Done when:** the TUI runs live on a real GPU, and the README has the benchmark table, architecture diagram and demo capture.

---

## 6. Benchmark rules
- Run the first GPU call once as a warm-up (it includes JIT compile time) and don't count it; report compile time separately.
- Call `cuda.synchronize()` before stopping a GPU timer, because GPU calls return before the work finishes.
- Use `time.perf_counter()`, repeat each run 3 times, and report the median.
- Always compare on the same input, on the same machine.

## 7. Results log (fill in)

| Input | Backend | Time | Bases/sec | Speedup vs pure Python |
|---|---|---|---|---|
| chr21 slice (5 M) | Pure Python (CPU) | | | 1× |
| chr21 slice (5 M) | Numba CPU | | | |
| chr21 slice (5 M) | GPU, 1× T4 | | | |
| chr21 (46 M) | GPU, 1× T4 | | | |
| hg38 (3.1 G) | GPU, 1× T4 | | | |
| hg38 (3.1 G) | GPU, 2× T4 (Dask) | | | |
| hg38 (3.1 G) | GPU, 2× T4 + shared memory | | | |

---

## 8. Glossary
- **Guide RNA:** the 20-letter sequence that tells Cas9 where to cut.
- **PAM:** a short tag (`NGG` for SpCas9) that must sit right after the target, or Cas9 won't cut there.
- **Off-target site:** a place in the genome that is *almost* the target (a few mismatches) where Cas9 might cut by mistake.
- **Reverse complement:** the other DNA strand, read backwards with A↔T and C↔G swapped.
- **Kernel:** a function that runs on the GPU, with thousands of copies (threads) running at once.
- **Host / Device:** host = CPU and its RAM; device = GPU and its memory (VRAM). Data must be copied between them.
- **JIT (Just-In-Time):** Numba turns your Python function into GPU machine code the first time you call it.
- **Shared memory:** a small, very fast memory that all threads in one block can share.
- **Dask worker:** a separate process that does part of the work. With `LocalCUDACluster`, there's one per GPU.
