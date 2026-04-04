"""
model/transformer.py  (JAX / Flax)
=====================================
Inverse-folding transformer rewritten in Flax.

PyTorch → JAX/Flax translation map
-------------------------------------

  PyTorch                        JAX / Flax
  ─────────────────────────────────────────────────────────────
  nn.Module                   →  flax.linen.Module
  nn.Linear                   →  nn.Dense
  nn.Embedding                →  nn.Embed
  nn.LayerNorm                →  nn.LayerNorm
  nn.GELU()                   →  nn.activation.gelu
  F.softmax                   →  jax.nn.softmax
  F.log_softmax               →  jax.nn.log_softmax
  torch.einsum                →  jnp.einsum
  model.parameters()          →  flax.traverse_util / optax
  model.train() / eval()      →  deterministic=True/False flag in forward
  torch.no_grad()             →  not needed (JAX is grad-free by default)
  model(x)                    →  model.apply(params, x)

Static shape requirement
------------------------
JAX / XLA requires statically shaped arrays for jit compilation.
All CDR-H3 token sequences are padded to MAX_CDR_LEN=28 with PAD_ID=0.
A boolean causal+padding mask is computed inside each forward call.

Flax module pattern
-------------------
Flax modules are stateless — parameters are stored in an external pytree
dict called ``params``.  Calling the model looks like:

    params = model.init(key, token_ids, ag_coords, ag_types, ag_sasa,
                        row_ptr, col_idx, valid_mask, lengths)["params"]
    log_probs = model.apply({"params": params}, token_ids, ...)
"""

from __future__ import annotations

import math
from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from ..kernels.sparse_cdr_attention import sparse_cdr_attention, MAX_CDR_LEN

AA_VOCAB_SIZE = 23   # PAD=0, BOS=1, EOS=2, AA tokens 3-22


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal positional encoding.

    In Flax, sub-modules are declared as class attributes with type
    annotations.  Parameters computed from the architecture (like PE) are
    not learnable and are computed in ``__call__`` rather than stored.
    """
    d_model: int
    max_len: int = 512

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: [L, d_model] → x + PE[:L]"""
        L      = x.shape[0]
        pos    = jnp.arange(L)[:, None].astype(jnp.float32)         # [L, 1]
        div    = jnp.exp(
            jnp.arange(0, self.d_model, 2).astype(jnp.float32)
            * (-math.log(10000.0) / self.d_model)
        )                                                             # [d/2]
        sin_pe = jnp.sin(pos * div)                                  # [L, d/2]
        cos_pe = jnp.cos(pos * div)                                  # [L, d/2]
        # Interleave sin / cos
        pe = jnp.reshape(
            jnp.stack([sin_pe, cos_pe], axis=-1),
            (L, self.d_model)
        )
        return x + pe


class EpitopePairEncoder(nn.Module):
    """
    MLP encoder for antigen backbone coordinates → node embeddings.

    Input : coords [n_ag, 3], aa_types [n_ag] int, sasa [n_ag] float
    Output: [n_ag, d_model]
    """
    d_model: int
    n_aa:    int = 20

    @nn.compact
    def __call__(
        self,
        coords:   jnp.ndarray,    # [n_ag, 3]
        aa_types: jnp.ndarray,    # [n_ag]   int32
        sasa:     jnp.ndarray,    # [n_ag]   float32
    ) -> jnp.ndarray:
        aa_emb = nn.Embed(num_embeddings=self.n_aa, features=32)(aa_types)   # [n_ag, 32]
        x = jnp.concatenate([coords, aa_emb, sasa[:, None]], axis=-1)        # [n_ag, 35+]
        x = nn.Dense(self.d_model)(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model)(x)
        return x


class CDRPositionEmbedding(nn.Module):
    """
    Token + sinusoidal position embedding for partial CDR-H3 sequences.

    Input : token_ids [MAX_CDR_LEN] int32 (padded)
    Output: [MAX_CDR_LEN, d_model]
    """
    vocab_size: int
    d_model:    int
    max_len:    int = MAX_CDR_LEN + 4

    @nn.compact
    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        # token_ids: [MAX_CDR_LEN]
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(token_ids)
        x = SinusoidalPositionEmbedding(d_model=self.d_model, max_len=self.max_len)(x)
        return x


class InverseFoldingLayer(nn.Module):
    """
    One transformer layer:
      1. Causal self-attention (CDR)
      2. Sparse cross-attention (CDR ← epitope-adjacent antigen; Pallas/JAX)
      3. Feed-forward network

    Flax note: dropout requires a ``deterministic`` flag and a PRNG key
    passed via ``self.make_rng('dropout')``.
    """
    d_model:      int
    n_heads:      int
    ffn_dim:      int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        cdr_emb:      jnp.ndarray,    # [MAX_CDR_LEN, d_model]
        ag_emb:       jnp.ndarray,    # [n_ag,        d_model]
        row_ptr:      jnp.ndarray,    # [MAX_CDR_LEN + 1]  int32
        col_idx:      jnp.ndarray,    # [nnz]               int32
        valid_mask:   jnp.ndarray,    # [MAX_CDR_LEN]        bool
        lengths:      jnp.ndarray,    # [] scalar int32 — actual CDR length
        deterministic: bool = True,
    ) -> jnp.ndarray:

        head_dim  = self.d_model // self.n_heads
        max_cdr   = cdr_emb.shape[0]
        scale     = 1.0 / math.sqrt(head_dim)

        # ── 1. Causal self-attention ──────────────────────────────────────
        def split_heads(x: jnp.ndarray) -> jnp.ndarray:
            """[L, D] → [L, H, Dh]"""
            return x.reshape(x.shape[0], self.n_heads, head_dim)

        def merge_heads(x: jnp.ndarray) -> jnp.ndarray:
            """[L, H, Dh] → [L, D]"""
            return x.reshape(x.shape[0], self.d_model)

        Q = split_heads(nn.Dense(self.d_model, use_bias=False)(cdr_emb))
        K = split_heads(nn.Dense(self.d_model, use_bias=False)(cdr_emb))
        V = split_heads(nn.Dense(self.d_model, use_bias=False)(cdr_emb))

        # Attention scores [H, max_cdr, max_cdr]
        attn = jnp.einsum("ihd,jhd->hij", Q, K) * scale

        # Causal mask — upper triangle = -inf
        causal = jnp.triu(
            jnp.full((max_cdr, max_cdr), -1e9), k=1
        )[None, :, :]
        # Padding mask — invalid rows/cols = -inf
        pad_mask = jnp.where(valid_mask, 0.0, -1e9)
        attn = attn + causal + pad_mask[None, :, None] + pad_mask[None, None, :]

        attn_w   = jax.nn.softmax(attn, axis=-1)   # [H, max_cdr, max_cdr]
        self_out = jnp.einsum("hij,jhd->ihd", attn_w, V)
        self_out = nn.Dense(self.d_model, use_bias=False)(merge_heads(self_out))
        self_out = nn.Dropout(rate=self.dropout_rate)(self_out, deterministic=deterministic)
        cdr_emb  = nn.LayerNorm()(cdr_emb + self_out)

        # ── 2. Sparse cross-attention (Pallas/JAX kernel) ─────────────────
        Qc = split_heads(nn.Dense(self.d_model, use_bias=False)(cdr_emb))
        Kc = split_heads(nn.Dense(self.d_model, use_bias=False)(ag_emb))
        Vc = split_heads(nn.Dense(self.d_model, use_bias=False)(ag_emb))

        cross_out = sparse_cdr_attention(
            Qc, Kc, Vc, row_ptr, col_idx, valid_mask
        )  # [max_cdr, H, Dh]
        cross_out = nn.Dense(self.d_model, use_bias=False)(merge_heads(cross_out))
        cross_out = nn.Dropout(rate=self.dropout_rate)(cross_out, deterministic=deterministic)
        cdr_emb   = nn.LayerNorm()(cdr_emb + cross_out)

        # ── 3. Feed-forward ───────────────────────────────────────────────
        ffn_out  = nn.Dense(self.ffn_dim)(cdr_emb)
        ffn_out  = nn.gelu(ffn_out)
        ffn_out  = nn.Dropout(rate=self.dropout_rate)(ffn_out, deterministic=deterministic)
        ffn_out  = nn.Dense(self.d_model)(ffn_out)
        ffn_out  = nn.Dropout(rate=self.dropout_rate)(ffn_out, deterministic=deterministic)
        cdr_emb  = nn.LayerNorm()(cdr_emb + ffn_out)

        return cdr_emb


class InverseFoldingTransformer(nn.Module):
    """
    Full inverse-folding transformer — Flax implementation.

    Usage (Flax functional API)
    ---------------------------
    model  = InverseFoldingTransformer(d_model=256, n_heads=8, n_layers=6,
                                       ffn_dim=1024)
    key    = jax.random.PRNGKey(0)
    params = model.init(key, token_ids, ag_coords, ag_types, ag_sasa,
                        row_ptr, col_idx, valid_mask, lengths)["params"]

    # Inference (no dropout)
    log_probs = model.apply({"params": params}, token_ids, ag_coords,
                            ag_types, ag_sasa, row_ptr, col_idx,
                            valid_mask, lengths)

    # Training (with dropout)
    log_probs = model.apply(
        {"params": params},
        token_ids, ag_coords, ag_types, ag_sasa,
        row_ptr, col_idx, valid_mask, lengths,
        deterministic=False,
        rngs={"dropout": dropout_key},
    )
    """
    d_model:      int   = 256
    n_heads:      int   = 8
    n_layers:     int   = 6
    ffn_dim:      int   = 1024
    vocab_size:   int   = AA_VOCAB_SIZE
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        token_ids:    jnp.ndarray,    # [MAX_CDR_LEN]        int32 padded
        ag_coords:    jnp.ndarray,    # [n_ag, 3]            float32
        ag_types:     jnp.ndarray,    # [n_ag]               int32
        ag_sasa:      jnp.ndarray,    # [n_ag]               float32
        row_ptr:      jnp.ndarray,    # [MAX_CDR_LEN + 1]    int32
        col_idx:      jnp.ndarray,    # [nnz]                int32
        valid_mask:   jnp.ndarray,    # [MAX_CDR_LEN]         bool
        lengths:      jnp.ndarray,    # []  scalar int32
        deterministic: bool = True,
    ) -> jnp.ndarray:                 # [MAX_CDR_LEN, vocab_size]

        # Embeddings
        cdr_emb = CDRPositionEmbedding(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
        )(token_ids)                                          # [MAX_CDR_LEN, D]

        ag_emb = EpitopePairEncoder(
            d_model=self.d_model,
        )(ag_coords, ag_types, ag_sasa)                       # [n_ag, D]

        # Transformer layers
        for _ in range(self.n_layers):
            cdr_emb = InverseFoldingLayer(
                d_model      = self.d_model,
                n_heads      = self.n_heads,
                ffn_dim      = self.ffn_dim,
                dropout_rate = self.dropout_rate,
            )(cdr_emb, ag_emb, row_ptr, col_idx,
              valid_mask, lengths, deterministic=deterministic)

        # Token head
        logits    = nn.Dense(self.vocab_size)(cdr_emb)        # [MAX_CDR_LEN, V]
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return log_probs
