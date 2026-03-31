# AntibodyForge

**De novo CDR-H3 loop design engine powered by inverse folding.**

AntibodyForge takes an antibody–antigen complex structure (PDB file) and
designs new CDR-H3 sequences predicted to bind the target epitope.
It is built on JAX, Flax, Optax, and a custom Pallas sparse attention kernel,
with a Rust slab allocator backing the beam search for zero GC overhead.

---

## What it does

Given a PDB file containing an antibody heavy chain and an antigen, AntibodyForge:

1. Detects the CDR-H3 loop automatically (Kabat, IMGT, or Chothia numbering)
2. Computes the epitope adjacency graph between CDR positions and antigen residues
3. Runs an inverse-folding transformer to score amino acid sequences conditioned on the antigen structure
4. Uses beam search to find the top-k CDR-H3 sequences with the lowest predicted binding free energy

---

## Architecture

```
PDB file
   │
   ├─ CDR-H3 detection (Kabat / IMGT / Chothia)
   ├─ Epitope adjacency graph  →  CSR tensors (row_ptr, col_idx, valid_mask)
   │
   ▼
InverseFoldingTransformer  (Flax)
   ├─ CDRPositionEmbedding      token + sinusoidal position
   ├─ EpitopePairEncoder        MLP over antigen Cα coords + AA type + SASA
   └─ N × InverseFoldingLayer
         ├─ Causal self-attention    (CDR positions)
         └─ Sparse cross-attention   (CDR ← epitope-adjacent antigen)
               └─ Pallas kernel over CSR adjacency  →  4× VRAM reduction
   │
   ▼
BeamSearchEngine
   ├─ Rust slab allocator  →  zero Python GC overhead for 10 000-wide beam
   └─ GC-free ΔG scorer    →  pre-allocated numpy buffer, no Python floats
   │
   ▼
Top-k CDR-H3 sequences  →  FASTA output
```

### Three core engineering decisions

**Sparse Pallas attention kernel**
The cross-attention between CDR-H3 positions and antigen residues uses a
Pallas kernel operating directly over a CSR-format epitope adjacency.
Only epitope-adjacent antigen residues (~10–20%) are attended to, cutting
VRAM by 4× for a 28-residue CDR-H3 compared to dense attention.

**Rust slab allocator for beam states**
Beam search at width 10 000 and depth 28 would naively create hundreds of
thousands of Python objects per antibody design. Instead every beam
hypothesis is stored as a flat integer array inside a Rust slab allocator.
Python holds only an integer handle. Pruned slots are reclaimed immediately
with no GC involvement.

**GC-free ΔG scoring**
The surrogate binding free energy scorer (CFFI-wrapped Rosetta FastRelax)
writes results directly into a pre-allocated `numpy.float32` buffer via a
C pointer. No Python float objects are created inside the beam loop across
all 50 steps × 1 000 candidates per antibody design.

---

## Repository structure

```
antibodyforge/
├── src/
│   ├── kernels/
│   │   └── sparse_cdr_attention.py   # Pallas GPU kernel + pure-JAX fallback
│   ├── beam/
│   │   ├── slab.rs                   # Rust slab allocator (C FFI)
│   │   ├── slab_allocator.py         # Python ctypes bridge
│   │   └── search.py                 # Beam search engine
│   ├── scorer/
│   │   └── delta_g.py                # GC-free ΔG scorer
│   ├── model/
│   │   └── transformer.py            # Flax inverse-folding transformer
│   ├── utils/
│   │   └── structure.py              # PDB parsing and epitope adjacency
│   └── pipeline.py                   # End-to-end design pipeline
├── scripts/
│   ├── train.py                      # Training CLI (Optax + Orbax)
│   └── design.py                     # Inference CLI
├── tests/
│   ├── unit/                         # Per-component unit tests
│   └── integration/                  # End-to-end pipeline tests
├── benchmarks/
│   └── vram_gc_benchmark.py          # VRAM, GC pressure, JIT warmup benchmarks
├── configs/
│   └── default.yaml                  # Default hyperparameters
├── Cargo.toml                        # Rust slab crate
└── pyproject.toml                    # Python package
```

---

## Requirements

| Dependency | Version | Role |
|---|---|---|
| Python | ≥ 3.10 | |
| JAX (CUDA) | ≥ 0.4.24 | Framework + Pallas kernels |
| Flax | ≥ 0.8.0 | Neural network modules |
| Optax | ≥ 0.2.0 | Optimisers and learning rate schedules |
| Orbax | ≥ 0.5.0 | Checkpoint saving and loading |
| NumPy | ≥ 1.26 | Score buffer (framework-agnostic) |
| Rust | ≥ 1.78 | Slab allocator build |
| Rosetta | 3.x | Optional — mock mode used if absent |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/antibodyforge.git
cd antibodyforge

# 2. Build the Rust slab allocator
cargo build --release
# produces: target/release/libbeam_slab.so

# 3. Install Python dependencies (GPU)
pip install -e ".[dev,train]"

# 3b. CPU-only install
pip install -e ".[cpu,dev]"
```

---

## Usage

### Design CDR-H3 sequences

```bash
python scripts/design.py \
    --pdb            data/target_complex.pdb \
    --heavy-chain    H \
    --antigen-chains A B \
    --checkpoint     checkpoints/final \
    --top-k          10 \
    --output         results/top10.fasta
```

### Python API

```python
from src.pipeline import AntibodyForgePipeline

pipeline = AntibodyForgePipeline.from_config(
    "configs/default.yaml",
    checkpoint_path="checkpoints/final",
)

results = pipeline.design(
    pdb_path          = "data/target_complex.pdb",
    heavy_chain_id    = "H",
    antigen_chain_ids = ["A"],
    top_k             = 10,
)

for r in results:
    print(f"{r.sequence}   logP={r.logprob:.3f}   ΔG={r.delta_g:.2f} kcal/mol")
```

### Training

```bash
python scripts/train.py \
    --config     configs/default.yaml \
    --data       data/train.jsonl \
    --val-data   data/val.jsonl \
    --epochs     100 \
    --output-dir checkpoints/
```

Training data is a JSON Lines file. Each line:

```json
{
  "pdb":            "data/structures/1ahw.pdb",
  "heavy_chain":    "H",
  "antigen_chains": ["A"],
  "cdr_h3_seq":     "ARDYYYYYGMDV"
}
```

---

## Configuration

All settings live in `configs/default.yaml`:

```yaml
# Model
d_model:      256
n_heads:      8
n_layers:     6
ffn_dim:      1024
dropout_rate: 0.1

# Beam search
beam_width:     10000
max_cdr_len:    28
length_penalty: 0.6
top_k_results:  10
n_slab_slots:   300000

# Scorer
max_candidates:    1000
n_rosetta_threads: 8

# Structure
epitope_cutoff_A: 6.0
numbering_scheme: kabat   # kabat | imgt | chothia

# JAX
seed: 42
```

---

## Running tests

```bash
# Unit tests (no GPU or Rosetta required)
pytest tests/unit/

# Integration tests
pytest tests/integration/

# All tests with coverage
pytest --cov=src tests/
```

---

## Benchmarks

```bash
python benchmarks/vram_gc_benchmark.py
```

Three benchmarks run automatically:

- **VRAM** — sparse vs dense CDR attention memory footprint across CDR lengths 7–28
- **GC pressure** — Python float allocations: naive scorer vs pre-allocated buffer
- **JIT warmup** — JAX compilation cost on first call vs steady-state kernel cache

---

## CDR-H3 numbering schemes

| Scheme | Residue range |
|---|---|
| Kabat (default) | 95 – 102 |
| IMGT | 105 – 117 |
| Chothia | 95 – 102 |

If no residues fall in the numbered range, a loop-length heuristic
(Cys–Trp bracket, 7–28 residues) is used as fallback.

---

## Output format

Results are written as FASTA with score annotations:

```
>design_001 logprob=-1.2340 dG=-14.72
ARDYYYYYGMDV
>design_002 logprob=-1.3891 dG=-13.45
ARDSSSYYGMDV
```

---

## License

Apache 2.0
