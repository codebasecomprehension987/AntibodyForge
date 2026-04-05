"""
tests/unit/test_sparse_attention.py  (JAX)
==========================================
Unit tests for the JAX/Pallas sparse CDR attention kernel.
Falls back to the pure-JAX reference implementation on CPU.
"""

from __future__ import annotations

import math
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from src.kernels.sparse_cdr_attention import (
    build_csr_adjacency,
    sparse_cdr_attention_reference,
    MAX_CDR_LEN,
)


# ---------------------------------------------------------------------------
# Dense reference for validation
# ---------------------------------------------------------------------------

def dense_attention_full(Q, K, V):
    """Full dense attention [n_cdr, H, Dh] over all antigen residues."""
    scale  = 1.0 / math.sqrt(Q.shape[-1])
    scores = jnp.einsum("ihd,jhd->hij", Q, K) * scale   # [H, n_cdr, n_ag]
    w      = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("hij,jhd->ihd", w, V)             # [n_cdr, H, Dh]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_random_csr(n_cdr, n_ag, density, seed=0):
    rng   = np.random.default_rng(seed)
    edges = []
    for r in range(n_cdr):
        for c in range(n_ag):
            if rng.random() < density:
                edges.append((r, c))
        if not any(e[0] == r for e in edges):
            edges.append((r, 0))
    return build_csr_adjacency(n_cdr, n_ag, edges)


# ---------------------------------------------------------------------------
# Tests: build_csr_adjacency
# ---------------------------------------------------------------------------

class TestBuildCsrAdjacency:

    def test_empty_edges(self):
        rp, ci, vm = build_csr_adjacency(3, 5, [])
        assert rp.shape == (MAX_CDR_LEN + 1,)
        assert ci.shape == (0,)
        assert vm[:3].tolist() == [True, True, True]
        assert vm[3:].tolist() == [False] * (MAX_CDR_LEN - 3)

    def test_single_edge(self):
        rp, ci, _ = build_csr_adjacency(2, 3, [(0, 2)])
        assert rp[0] == 0
        assert rp[1] == 1
        assert ci[0] == 2

    def test_valid_mask_length(self):
        _, _, vm = build_csr_adjacency(7, 10, [(0, 0)])
        assert vm.shape == (MAX_CDR_LEN,)
        assert vm[:7].all()
        assert not vm[7:].any()

    def test_dtypes(self):
        rp, ci, vm = build_csr_adjacency(2, 4, [(0, 1)])
        assert rp.dtype == np.int32
        assert ci.dtype == np.int32
        assert vm.dtype == bool


# ---------------------------------------------------------------------------
# Tests: sparse_cdr_attention_reference
# ---------------------------------------------------------------------------

class TestSparseCdrAttentionReference:

    def _make_inputs(self, n_cdr, n_ag, H, Dh, density=0.2, seed=0):
        key = jax.random.PRNGKey(seed)
        k1, k2, k3 = jax.random.split(key, 3)
        Q  = jax.random.normal(k1, (MAX_CDR_LEN, H, Dh))
        K  = jax.random.normal(k2, (n_ag, H, Dh))
        V  = jax.random.normal(k3, (n_ag, H, Dh))
        rp, ci, vm = make_random_csr(n_cdr, n_ag, density, seed=seed)
        return (
            Q, K, V,
            jnp.array(rp), jnp.array(ci), jnp.array(vm),
        )

    def test_full_adjacency_matches_dense(self):
        """When all pairs connected, sparse ref must equal dense attention."""
        n_cdr, n_ag, H, Dh = 7, 5, 2, 16
        key = jax.random.PRNGKey(1)
        k1, k2, k3 = jax.random.split(key, 3)
        Q = jax.random.normal(k1, (MAX_CDR_LEN, H, Dh))
        K = jax.random.normal(k2, (n_ag, H, Dh))
        V = jax.random.normal(k3, (n_ag, H, Dh))

        edges = [(r, c) for r in range(n_cdr) for c in range(n_ag)]
        rp, ci, vm = build_csr_adjacency(n_cdr, n_ag, edges)

        sparse_out = sparse_cdr_attention_reference(
            Q, K, V,
            jnp.array(rp), jnp.array(ci), jnp.array(vm),
        )
        # Dense attention over only valid rows
        dense_out = dense_attention_full(Q[:n_cdr], K, V)  # [n_cdr, H, Dh]

        np.testing.assert_allclose(
            np.array(sparse_out[:n_cdr]),
            np.array(dense_out),
            atol=1e-5,
        )

    def test_padding_rows_are_zero(self):
        """Padded CDR rows (valid_mask=False) must produce zero output."""
        n_cdr, n_ag, H, Dh = 7, 10, 2, 16
        Q, K, V, rp, ci, vm = self._make_inputs(n_cdr, n_ag, H, Dh)
        out = sparse_cdr_attention_reference(Q, K, V, rp, ci, vm)
        # Rows beyond n_cdr should be zero
        assert jnp.all(out[n_cdr:] == 0.0)

    def test_output_shape(self):
        n_cdr, n_ag, H, Dh = 14, 40, 4, 32
        Q, K, V, rp, ci, vm = self._make_inputs(n_cdr, n_ag, H, Dh)
        out = sparse_cdr_attention_reference(Q, K, V, rp, ci, vm)
        assert out.shape == (MAX_CDR_LEN, H, Dh)

    def test_jit_compatible(self):
        """sparse_cdr_attention_reference must be JIT-compilable."""
        n_cdr, n_ag, H, Dh = 7, 10, 2, 16
        Q, K, V, rp, ci, vm = self._make_inputs(n_cdr, n_ag, H, Dh)
        jit_fn = jax.jit(sparse_cdr_attention_reference)
        out    = jit_fn(Q, K, V, rp, ci, vm)
        assert out.shape == (MAX_CDR_LEN, H, Dh)

    @pytest.mark.parametrize("n_cdr,n_ag,H,Dh", [
        (7,  20, 4, 32),
        (14, 60, 8, 64),
        (28, 120, 8, 64),
    ])
    def test_various_sizes(self, n_cdr, n_ag, H, Dh):
        Q, K, V, rp, ci, vm = self._make_inputs(n_cdr, n_ag, H, Dh, density=0.15)
        out = sparse_cdr_attention_reference(Q, K, V, rp, ci, vm)
        assert out.shape == (MAX_CDR_LEN, H, Dh)
        assert not jnp.any(jnp.isnan(out))
