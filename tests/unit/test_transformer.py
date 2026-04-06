"""
tests/unit/test_transformer.py  (JAX)
======================================
Unit tests for the Flax inverse-folding transformer.

Key JAX/Flax-specific tests (no equivalent in PyTorch version)
--------------------------------------------------------------
- model.init returns a pytree of params (not in-place state)
- model.apply is the only way to call the model (functional API)
- jax.jit compilation of the forward pass
- jax.grad differentiability through the model
- vmap over batch dimension
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.model.transformer import (
    InverseFoldingTransformer,
    EpitopePairEncoder,
    CDRPositionEmbedding,
    AA_VOCAB_SIZE,
)
from src.kernels.sparse_cdr_attention import build_csr_adjacency, MAX_CDR_LEN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_AG     = 10
N_HEADS  = 2
D_MODEL  = 32
FFN_DIM  = 64
N_LAYERS = 2


def _dummy_inputs(n_ag=N_AG):
    """Return dummy model inputs as JAX arrays."""
    token_ids  = jnp.zeros(MAX_CDR_LEN, dtype=jnp.int32)
    token_ids  = token_ids.at[0].set(1)   # BOS
    ag_coords  = jnp.zeros((n_ag, 3),  dtype=jnp.float32)
    ag_types   = jnp.zeros(n_ag,        dtype=jnp.int32)
    ag_sasa    = jnp.zeros(n_ag,        dtype=jnp.float32)
    # Full adjacency: every CDR row connected to every antigen col
    edges      = [(r, c) for r in range(7) for c in range(n_ag)]
    rp, ci, vm = build_csr_adjacency(7, n_ag, edges)
    row_ptr    = jnp.array(rp, dtype=jnp.int32)
    col_idx    = jnp.array(ci, dtype=jnp.int32)
    valid_mask = jnp.array(vm, dtype=bool)
    lengths    = jnp.array(7,  dtype=jnp.int32)
    return token_ids, ag_coords, ag_types, ag_sasa, row_ptr, col_idx, valid_mask, lengths


def _make_model():
    return InverseFoldingTransformer(
        d_model      = D_MODEL,
        n_heads      = N_HEADS,
        n_layers     = N_LAYERS,
        ffn_dim      = FFN_DIM,
        dropout_rate = 0.0,
    )


# ---------------------------------------------------------------------------
# Tests: Flax API
# ---------------------------------------------------------------------------

class TestFlaxAPI:

    def test_init_returns_params_pytree(self):
        """model.init must return a dict (pytree), not a stateful object."""
        model  = _make_model()
        key    = jax.random.PRNGKey(0)
        inputs = _dummy_inputs()
        variables = model.init(key, *inputs)
        assert isinstance(variables, dict)
        assert "params" in variables

    def test_apply_returns_log_probs(self):
        model  = _make_model()
        key    = jax.random.PRNGKey(0)
        inputs = _dummy_inputs()
        params = model.init(key, *inputs)["params"]
        out    = model.apply({"params": params}, *inputs)
        assert out.shape == (MAX_CDR_LEN, AA_VOCAB_SIZE)

    def test_log_probs_sum_to_one(self):
        model  = _make_model()
        key    = jax.random.PRNGKey(0)
        inputs = _dummy_inputs()
        params = model.init(key, *inputs)["params"]
        out    = model.apply({"params": params}, *inputs)   # [MAX_CDR_LEN, V]
        # Each position's probabilities must sum to 1
        probs  = jnp.exp(out)
        np.testing.assert_allclose(
            np.array(probs.sum(axis=-1)),
            np.ones(MAX_CDR_LEN),
            atol=1e-5,
        )

    def test_params_are_pytree(self):
        """Params must be a nested dict (pytree), not nn.Parameter objects."""
        model  = _make_model()
        key    = jax.random.PRNGKey(0)
        params = model.init(key, *_dummy_inputs())["params"]
        # Should be traversable as a pytree
        leaves = jax.tree_util.tree_leaves(params)
        assert len(leaves) > 0
        for leaf in leaves:
            assert isinstance(leaf, jnp.ndarray)


class TestJitCompatibility:

    def test_jit_forward(self):
        """model.apply must be JIT-compilable."""
        model  = _make_model()
        key    = jax.random.PRNGKey(0)
        inputs = _dummy_inputs()
        params = model.init(key, *inputs)["params"]

        @jax.jit
        def forward(params, *inputs):
            return model.apply({"params": params}, *inputs)

        out = forward(params, *inputs)
        assert out.shape == (MAX_CDR_LEN, AA_VOCAB_SIZE)

    def test_jit_consistent(self):
        """JIT and non-JIT outputs must be identical."""
        model  = _make_model()
        key    = jax.random.PRNGKey(0)
        inputs = _dummy_inputs()
        params = model.init(key, *inputs)["params"]

        out_eager = model.apply({"params": params}, *inputs)
        out_jit   = jax.jit(lambda p: model.apply({"params": p}, *inputs))(params)

        np.testing.assert_allclose(
            np.array(out_eager), np.array(out_jit), atol=1e-6
        )


class TestDifferentiability:

    def test_grad_flows_through_model(self):
        """jax.grad must produce non-zero gradients for model parameters."""
        model  = _make_model()
        key    = jax.random.PRNGKey(0)
        inputs = _dummy_inputs()
        params = model.init(key, *inputs)["params"]

        def loss_fn(params):
            log_probs = model.apply({"params": params}, *inputs)
            # Simple sum loss
            return -log_probs[0, 1]   # log-prob of token 1 at position 0

        grads = jax.grad(loss_fn)(params)
        leaves = jax.tree_util.tree_leaves(grads)
        # At least some gradients must be non-zero
        has_nonzero = any(np.any(np.abs(np.array(g)) > 1e-8) for g in leaves)
        assert has_nonzero, "All gradients are zero — autodiff is broken"


class TestSubModules:

    def test_epitope_encoder_output_shape(self):
        enc    = EpitopePairEncoder(d_model=D_MODEL)
        key    = jax.random.PRNGKey(0)
        coords = jnp.zeros((N_AG, 3),  dtype=jnp.float32)
        types  = jnp.zeros(N_AG,        dtype=jnp.int32)
        sasa   = jnp.zeros(N_AG,        dtype=jnp.float32)
        params = enc.init(key, coords, types, sasa)["params"]
        out    = enc.apply({"params": params}, coords, types, sasa)
        assert out.shape == (N_AG, D_MODEL)

    def test_cdr_embedding_output_shape(self):
        emb    = CDRPositionEmbedding(vocab_size=AA_VOCAB_SIZE, d_model=D_MODEL)
        key    = jax.random.PRNGKey(0)
        tokens = jnp.zeros(MAX_CDR_LEN, dtype=jnp.int32)
        params = emb.init(key, tokens)["params"]
        out    = emb.apply({"params": params}, tokens)
        assert out.shape == (MAX_CDR_LEN, D_MODEL)
