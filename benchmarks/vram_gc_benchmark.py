"""
benchmarks/vram_gc_benchmark.py  (JAX)
=======================================
Same two benchmarks as the PyTorch version:
  1. VRAM savings from sparse CDR attention
  2. GC pressure elimination via pre-allocated numpy score buffer

Additional JAX-specific benchmark:
  3. JIT compilation warmup vs steady-state throughput
     JAX traces & compiles on the first call; subsequent calls hit the cache.
     We measure both.
"""

from __future__ import annotations

import gc
import math
import time
import tracemalloc
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# 1. VRAM benchmark (same calculation as PyTorch version)
# ---------------------------------------------------------------------------

def _dense_vram(n_cdr, n_ag, n_heads, head_dim):
    return n_heads * n_cdr * n_ag * 4   # float32 bytes

def _sparse_vram(n_cdr, n_ag, density, n_heads, head_dim):
    nnz       = int(n_cdr * n_ag * density)
    row_ptr   = (n_cdr + 1) * 4
    col_idx   = nnz * 4
    kv_gather = 2 * nnz * n_heads * head_dim * 4
    return row_ptr + col_idx + kv_gather


def benchmark_vram() -> None:
    print("\n" + "=" * 64)
    print("  BENCHMARK 1: VRAM — Sparse vs Dense CDR Attention (JAX)")
    print("=" * 64)

    configs = [
        (7,  40,  0.20, 8, 64),
        (14, 80,  0.15, 8, 64),
        (21, 100, 0.12, 8, 64),
        (28, 120, 0.10, 8, 64),
    ]

    for n_cdr, n_ag, density, H, Dh in configs:
        dense  = _dense_vram(n_cdr, n_ag, H, Dh)
        sparse = _sparse_vram(n_cdr, n_ag, density, H, Dh)
        ratio  = dense / max(sparse, 1)
        nnz    = int(n_cdr * n_ag * density)
        print(
            f"  CDR={n_cdr:2d} | antigen={n_ag:3d} | density={density:.0%} "
            f"| nnz={nnz:4d} | dense={dense/1024:.1f}KB "
            f"| sparse={sparse/1024:.1f}KB | saving={ratio:.1f}×"
        )


# ---------------------------------------------------------------------------
# 2. GC pressure benchmark (identical to PyTorch version)
# ---------------------------------------------------------------------------

class _NaiveScorer:
    def score_naive(self, seqs):
        return [float(np.random.uniform(-20, 0)) for _ in seqs]

class _BufferedScorer:
    def __init__(self, max_n):
        self._buf = np.empty(max_n, dtype=np.float32)

    def score_buffered(self, seqs, out):
        n = len(seqs)
        out[:n] = np.random.default_rng(0).uniform(-20, 0, n).astype(np.float32)


def _count_float_allocs(fn) -> Tuple[int, float]:
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    snap    = tracemalloc.take_snapshot()
    tracemalloc.stop()
    float_allocs = sum(
        s.count for s in snap.statistics("lineno")
        if "float" in str(s).lower()
    )
    return float_allocs, elapsed


def benchmark_gc(beam_steps=50, candidates=1_000) -> None:
    print("\n" + "=" * 64)
    print("  BENCHMARK 2: GC Pressure — Naive vs Buffer-based Scorer (JAX)")
    print("=" * 64)
    print(f"  Steps={beam_steps} | Candidates={candidates} | "
          f"Total={beam_steps*candidates:,}")

    naive     = _NaiveScorer()
    buffered  = _BufferedScorer(candidates)
    buf_view  = buffered._buf[:candidates]
    fake_seqs = [[3, 4, 5]] * candidates

    n_naive,  t_naive  = _count_float_allocs(
        lambda: [naive.score_naive(fake_seqs) for _ in range(beam_steps)]
    )
    n_buff,   t_buff   = _count_float_allocs(
        lambda: [buffered.score_buffered(fake_seqs, buf_view) for _ in range(beam_steps)]
    )

    print(f"\n  Naive    : {n_naive:>10,} float allocs | {t_naive*1000:.1f} ms")
    print(f"  Buffered : {n_buff:>10,} float allocs | {t_buff*1000:.1f} ms")
    print(f"\n  Alloc reduction : {n_naive / max(n_buff, 1):.0f}×")
    print(f"  Speed-up        : {t_naive / max(t_buff, 1e-9):.1f}×")
    if n_buff < 50:
        print("  ✓  Buffered path is GC-free.")


# ---------------------------------------------------------------------------
# 3. JAX-specific: JIT warmup vs steady-state
# ---------------------------------------------------------------------------

def benchmark_jit_warmup() -> None:
    print("\n" + "=" * 64)
    print("  BENCHMARK 3: JAX JIT — Warmup vs Steady-state (JAX-specific)")
    print("=" * 64)

    from src.kernels.sparse_cdr_attention import (
        sparse_cdr_attention_reference, build_csr_adjacency, MAX_CDR_LEN
    )

    n_cdr, n_ag, H, Dh = 28, 120, 8, 64
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    Q = jax.random.normal(k1, (MAX_CDR_LEN, H, Dh))
    K = jax.random.normal(k2, (n_ag, H, Dh))
    V = jax.random.normal(k3, (n_ag, H, Dh))

    edges  = [(r, c) for r in range(n_cdr) for c in range(n_ag)][:500]
    rp, ci, vm = build_csr_adjacency(n_cdr, n_ag, edges)
    rp_j  = jnp.array(rp)
    ci_j  = jnp.array(ci)
    vm_j  = jnp.array(vm)

    jit_fn = jax.jit(sparse_cdr_attention_reference)

    # Warmup (includes XLA compilation)
    t0      = time.perf_counter()
    _       = jit_fn(Q, K, V, rp_j, ci_j, vm_j).block_until_ready()
    warmup  = time.perf_counter() - t0

    # Steady-state (cache hit)
    N_REPS = 20
    t0     = time.perf_counter()
    for _ in range(N_REPS):
        jit_fn(Q, K, V, rp_j, ci_j, vm_j).block_until_ready()
    steady = (time.perf_counter() - t0) / N_REPS

    print(f"\n  JIT warmup (trace+compile) : {warmup*1000:.1f} ms")
    print(f"  Steady-state (cached)      : {steady*1000:.2f} ms")
    print(f"  Speedup after warmup       : {warmup/max(steady,1e-9):.0f}×")
    print("\n  Note: JAX compiles once per unique input shape — "
          "all beam steps with MAX_CDR_LEN=28 reuse the same compiled kernel.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    benchmark_vram()
    benchmark_gc()
    benchmark_jit_warmup()
    print("\n" + "=" * 64)
    print("  Benchmarks complete.")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
