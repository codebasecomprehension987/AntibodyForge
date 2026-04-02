"""
kernels/sparse_cdr_attention.py  (JAX / Pallas)
================================================
Pallas kernel for ragged-CSR sparse attention over CDR-H3 / epitope pairs.

JAX vs PyTorch kernel differences
-----------------------------------
In the PyTorch version we used Triton directly.  In JAX we use **Pallas**
(jax.experimental.pallas), which is JAX's native kernel-writing language
built on top of Triton but integrated with JAX's tracing, JIT, and
autodiff systems.

Key differences from the PyTorch/Triton version:
  - Pallas kernels are written as pure functions (no side effects)
  - Shape information must be statically known at trace time — we pad
    CDR-H3 sequences to MAX_CDR_LEN=28 and carry a valid-row mask
  - Pallas uses ``pl.load`` / ``pl.store`` instead of ``tl.load`` / ``tl.store``
  - The kernel is JIT-compiled via ``jax.jit`` automatically

Static shape constraint
------------------------
JAX requires static shapes for XLA compilation.  Variable-length CDR-H3
loops (7–28 residues) are handled by:
  1. Padding queries to shape [MAX_CDR_LEN, n_heads, head_dim]
  2. Carrying a boolean mask [MAX_CDR_LEN] indicating valid rows
  3. The CSR row_ptr still encodes per-row neighbour counts; padded rows
     have row_ptr[r] == row_ptr[r+1] (empty slice) so they produce zero output

Fallback: pure JAX scatter/gather attention
-------------------------------------------
When Pallas is unavailable (CPU-only environments, older JAX versions) we
provide a pure JAX reference implementation using ``jnp.take`` for gathering
and ``jax.ops.segment_sum`` (via ``jax.lax.associative_scan``) for the
softmax accumulation.  The fallback is fully differentiable and JIT-able.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Pallas may not be available on all JAX builds; graceful fallback
try:
    import jax.experimental.pallas as pl
    import jax.experimental.pallas.ops as plops
    _PALLAS_AVAILABLE = True
except ImportError:
    _PALLAS_AVAILABLE = False

MAX_CDR_LEN = 28


# ---------------------------------------------------------------------------
# Pure-JAX reference implementation (fallback + autodiff baseline)
# ---------------------------------------------------------------------------

def sparse_cdr_attention_reference(
    Q:       jnp.ndarray,   # [max_cdr, n_heads, head_dim]
    K:       jnp.ndarray,   # [n_ag,    n_heads, head_dim]
    V:       jnp.ndarray,   # [n_ag,    n_heads, head_dim]
    row_ptr: jnp.ndarray,   # [max_cdr + 1]  int32
    col_idx: jnp.ndarray,   # [nnz]           int32
    valid_mask: jnp.ndarray,# [max_cdr]        bool
) -> jnp.ndarray:           # [max_cdr, n_heads, head_dim]
    """
    Pure-JAX sparse cross-attention.  Fully differentiable, JIT-able.

    Uses vmap over CDR positions.  Each position gathers its neighbours,
    computes masked softmax, and accumulates weighted values.
    """
    max_cdr, n_heads, head_dim = Q.shape
    scale = 1.0 / jnp.sqrt(head_dim).astype(jnp.float32)

    # Maximum neighbours any row could have = nnz (safe upper bound)
    max_nb = col_idx.shape[0]

    def attend_one_row(r: int):
        """Attend from CDR position r to its epitope neighbours."""
        start = row_ptr[r]
        end   = row_ptr[r + 1]
        # Gather neighbour indices (padded to max_nb with 0)
        nb_range = jnp.arange(max_nb)
        nb_mask  = nb_range < (end - start)
        # Safe gather: out-of-bounds indices clamped to 0
        nb_cols  = jnp.where(
            nb_mask,
            col_idx[jnp.clip(start + nb_range, 0, col_idx.shape[0] - 1)],
            0,
        )

        q = Q[r]                                   # [H, Dh]
        k = K[nb_cols]                             # [max_nb, H, Dh]
        v = V[nb_cols]                             # [max_nb, H, Dh]

        # Dot products: [H, max_nb]
        dots = jnp.einsum("hd,nhd->hn", q, k) * scale
        # Mask out padding
        dots = jnp.where(nb_mask[None, :], dots, -1e9)
        # Softmax
        w = jax.nn.softmax(dots, axis=-1)          # [H, max_nb]
        # Weighted sum: [H, Dh]
        out = jnp.einsum("hn,nhd->hd", w, v)
        # Zero out invalid CDR rows
        return jnp.where(valid_mask[r], out, jnp.zeros_like(out))

    # vmap over all CDR positions
    rows = jnp.arange(max_cdr)
    return jax.vmap(attend_one_row)(rows)          # [max_cdr, H, Dh]


# ---------------------------------------------------------------------------
# Pallas kernel (GPU fast path)
# ---------------------------------------------------------------------------

def _pallas_sparse_attn_kernel(
    Q_ref, K_ref, V_ref,
    row_ptr_ref, col_idx_ref, valid_ref,
    Out_ref,
    *,
    n_heads: int,
    head_dim: int,
    max_nb: int,
):
    """Pallas kernel body — one invocation per (cdr_row, head)."""
    r    = pl.program_id(0)
    head = pl.program_id(1)

    valid = pl.load(valid_ref, (r,))

    start = pl.load(row_ptr_ref, (r,))
    end   = pl.load(row_ptr_ref, (r + 1,))
    n_nb  = end - start

    q     = pl.load(Q_ref, (r, head, pl.dslice(0, head_dim)))  # [Dh]
    scale = 1.0 / jnp.sqrt(float(head_dim))
    q     = q * scale

    acc     = jnp.zeros(head_dim, dtype=jnp.float32)
    log_sum = jnp.array(-jnp.inf, dtype=jnp.float32)

    # Iterate over neighbours
    for i in range(max_nb):
        in_range = i < n_nb
        col  = pl.load(col_idx_ref, (start + i,))
        k    = pl.load(K_ref, (col, head, pl.dslice(0, head_dim)))
        v    = pl.load(V_ref, (col, head, pl.dslice(0, head_dim)))
        dot  = jnp.dot(q, k)
        dot  = jnp.where(in_range, dot, -jnp.inf)

        new_log_sum = jnp.logaddexp(log_sum, dot)
        acc         = acc * jnp.exp(log_sum - new_log_sum) + \
                      jnp.exp(dot - new_log_sum) * v
        log_sum     = new_log_sum

    acc = jnp.where(valid, acc, jnp.zeros_like(acc))
    pl.store(Out_ref, (r, head, pl.dslice(0, head_dim)), acc)


def sparse_cdr_attention_pallas(
    Q:          jnp.ndarray,
    K:          jnp.ndarray,
    V:          jnp.ndarray,
    row_ptr:    jnp.ndarray,
    col_idx:    jnp.ndarray,
    valid_mask: jnp.ndarray,
    max_nb:     int,
) -> jnp.ndarray:
    """Dispatch the Pallas sparse CDR attention kernel."""
    max_cdr, n_heads, head_dim = Q.shape

    out_shape = jax.ShapeDtypeStruct(Q.shape, Q.dtype)

    return pl.pallas_call(
        partial(
            _pallas_sparse_attn_kernel,
            n_heads=n_heads,
            head_dim=head_dim,
            max_nb=max_nb,
        ),
        out_shape=out_shape,
        grid=(max_cdr, n_heads),
    )(Q, K, V, row_ptr, col_idx, valid_mask)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def sparse_cdr_attention(
    Q:          jnp.ndarray,   # [max_cdr, n_heads, head_dim]
    K:          jnp.ndarray,   # [n_ag,    n_heads, head_dim]
    V:          jnp.ndarray,   # [n_ag,    n_heads, head_dim]
    row_ptr:    jnp.ndarray,   # [max_cdr + 1]  int32
    col_idx:    jnp.ndarray,   # [nnz]           int32
    valid_mask: jnp.ndarray,   # [max_cdr]        bool
    max_nb:     int = 64,
) -> jnp.ndarray:
    """
    Sparse CDR-H3 cross-attention.

    Dispatches to the Pallas GPU kernel when available, otherwise falls
    back to the pure-JAX reference implementation.

    Parameters
    ----------
    Q, K, V     : query/key/value arrays
    row_ptr     : CSR row pointer  [max_cdr + 1]
    col_idx     : CSR column indices [nnz]
    valid_mask  : True for real CDR positions, False for padding
    max_nb      : upper bound on neighbours per row (for static shapes)

    Returns
    -------
    jnp.ndarray : [max_cdr, n_heads, head_dim]
    """
    if _PALLAS_AVAILABLE and Q.device_buffers[0].platform() == "gpu":
        return sparse_cdr_attention_pallas(
            Q, K, V, row_ptr, col_idx, valid_mask, max_nb
        )
    return sparse_cdr_attention_reference(
        Q, K, V, row_ptr, col_idx, valid_mask
    )


# ---------------------------------------------------------------------------
# CSR builder (numpy — called once per protein pair, outside JIT)
# ---------------------------------------------------------------------------

def build_csr_adjacency(
    cdr_len:   int,
    n_antigen: int,
    edge_list: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build CSR adjacency arrays (numpy) from an edge list.

    Returns
    -------
    row_ptr    : int32 [MAX_CDR_LEN + 1]  — padded to MAX_CDR_LEN
    col_idx    : int32 [nnz]
    valid_mask : bool  [MAX_CDR_LEN]      — True for real rows
    """
    from collections import defaultdict

    adj: dict[int, list[int]] = defaultdict(list)
    for r, c in edge_list:
        adj[r].append(c)

    row_ptr_list = [0]
    col_idx_list = []
    for r in range(cdr_len):
        neighbours = sorted(adj.get(r, []))
        col_idx_list.extend(neighbours)
        row_ptr_list.append(len(col_idx_list))

    # Pad row_ptr to MAX_CDR_LEN + 1
    nnz = len(col_idx_list)
    for _ in range(MAX_CDR_LEN - cdr_len):
        row_ptr_list.append(nnz)   # empty padded rows

    valid_mask = np.array(
        [True] * cdr_len + [False] * (MAX_CDR_LEN - cdr_len), dtype=bool
    )
    row_ptr = np.array(row_ptr_list, dtype=np.int32)
    col_idx = np.array(col_idx_list, dtype=np.int32)
    return row_ptr, col_idx, valid_mask
