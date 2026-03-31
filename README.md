# AntibodyForge (JAX)

**De novo CDR-H3 inverse-folding engine** — Pallas sparse attention · Flax transformer · Optax training · Rust slab beam search · GC-free ΔG scoring

---

## What changed from the PyTorch version

| Component | PyTorch version | JAX version |
|---|---|---|
| Kernel language | Triton | **Pallas** (JAX's Triton wrapper) |
| Model framework | `torch.nn.Module` | **Flax** `nn.Module` |
| Parameter storage | In-module `nn.Parameter` | External pytree dict (`params`) |
| Optimiser | `torch.optim.AdamW` + `CosineAnnealingLR` | **Optax** `adamw` + `cosine_decay_schedule` |
| Checkpointing | `torch.save` / `torch.load` | **Orbax** `PyTreeCheckpointer` |
| Gradient computation | `loss.backward()` | `jax.value_and_grad(loss_fn)(params)` |
| JIT compilation | Implicit (CUDA kernels) | Explicit `jax.jit` |
| Batch dimension | Variable (dynamic shapes) | Static shapes padded to `MAX_CDR_LEN=28` |
| Random state | Global `torch.manual_seed` | Explicit `jax.random.PRNGKey` threading |
| Device placement | `tensor.to(device)` | `jax.default_device()` / env vars |
| **Rust slab** | Unchanged | **Identical** — framework-agnostic |
| **ΔG scorer** | Unchanged | **Identical** — numpy buffer, no JAX/PyTorch imports |

---

## Repository Structure

```
antibodyforge_jax/
├── src/
│   ├── kernels/
│   │   └── sparse_cdr_attention.py   # Pallas kernel + pure-JAX fallback + CSR builder
│   ├── beam/
│   │   ├── slab.rs                   # Rust slab allocator (identical to PyTorch version)
│   │   ├── slab_allocator.py         # ctypes bridge (identical to PyTorch version)
│   │   └── search.py                 # Beam search — jax.lax.top_k, jnp arrays
│   ├── scorer/
│   │   └── delta_g.py                # GC-free scorer (zero JAX/PyTorch imports)
│   ├── model/
│   │   └── transformer.py            # Flax InverseFoldingTransformer
│   ├── utils/
│   │   └── structure.py              # PDB parsing → jnp arrays + valid_mask
│   └── pipeline.py                   # ForgeConfig + AntibodyForgePipeline (JAX)
├── scripts/
│   ├── train.py                      # Optax training loop + Orbax checkpointing
│   └── design.py                     # Inference CLI
├── tests/
│   ├── unit/
│   │   ├── test_sparse_attention.py  # Pallas/JAX attention tests + JIT test
│   │   ├── test_beam_search.py       # jax.lax.top_k tests
│   │   ├── test_scorer.py            # GC-free buffer tests (same as PyTorch)
│   │   ├── test_structure.py         # jnp tensor output tests
│   │   └── test_transformer.py       # Flax API, autodiff, JIT tests (JAX-specific)
│   └── integration/
│       └── test_pipeline.py          # End-to-end test + JIT compilation test
├── benchmarks/
│   └── vram_gc_benchmark.py          # VRAM + GC + JAX JIT warmup benchmarks
├── configs/default.yaml
├── Cargo.toml
└── pyproject.toml
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/your-org/antibodyforge-jax.git
cd antibodyforge_jax

# 2. Build Rust slab allocator (unchanged from PyTorch version)
cargo build --release

# 3. Install Python package (GPU)
pip install -e ".[dev,train]"

# 3b. CPU-only
pip install -e ".[cpu,dev]"
```

---

## Quick Start

```python
from src.pipeline import AntibodyForgePipeline

pipeline = AntibodyForgePipeline.from_config("configs/default.yaml")
results  = pipeline.design(
    pdb_path          = "data/1ahw.pdb",
    heavy_chain_id    = "H",
    antigen_chain_ids = ["A"],
    top_k             = 10,
)
for r in results:
    print(r.sequence, r.logprob, r.delta_g)
```

---

## JAX-specific notes

### Static shapes
JAX / XLA requires statically shaped arrays for `jax.jit`.  All CDR-H3
sequences are padded to `MAX_CDR_LEN=28` with `PAD_ID=0`.  A `valid_mask`
boolean array marks real positions.  Padded rows produce zero output from
the attention kernel.

### Flax functional API
Parameters are stored outside the model:
```python
params    = model.init(key, *inputs)["params"]
log_probs = model.apply({"params": params}, *inputs)
```

### Optax training
```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule),
)
opt_state = optimizer.init(params)

(loss, _), grads  = jax.value_and_grad(loss_fn)(params, batch, key)
updates, opt_state = optimizer.update(grads, opt_state, params)
params             = optax.apply_updates(params, updates)
```

### JIT compilation
The model forward pass is compiled on the first call per unique input shape.
All beam search steps share `MAX_CDR_LEN=28` so compilation happens once
per antibody design run:
```python
jit_forward = jax.jit(lambda tokens, lengths: model.apply(...))
# First call: ~500ms compile + run
# Subsequent calls: ~2ms (cached XLA kernel)
```

### PRNG key management
```python
key, subkey = jax.random.split(key)
params = model.init(subkey, *dummy_inputs)["params"]
```

---

## Benchmarks

```bash
python benchmarks/vram_gc_benchmark.py
```

Includes three benchmarks:
1. VRAM savings (sparse vs dense CDR attention)
2. GC pressure (naive Python floats vs numpy buffer)
3. **JAX JIT warmup vs steady-state throughput** (JAX-specific)

---

## License

Apache 2.0
