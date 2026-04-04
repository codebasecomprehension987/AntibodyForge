"""
pipeline.py  (JAX)
===================
High-level AntibodyForge design pipeline — JAX rewrite.

Key differences from the PyTorch version
-----------------------------------------

1. **Flax functional API**
   ``model.apply({"params": params}, ...)`` instead of ``model(...)``
   Parameters are stored in an external pytree, not inside the module.

2. **Orbax checkpointing**
   ``orbax.checkpoint.PyTreeCheckpointer`` replaces ``torch.save/load``.
   Orbax is the canonical JAX checkpoint library, storing pytrees as
   sharded arrays or msgpack files.

3. **JAX PRNG key threading**
   JAX requires explicit PRNG key management.  The pipeline holds a
   ``jax.random.PRNGKey`` and splits it for each call that needs randomness
   (model init, dropout during training).

4. **Model function closure for beam search**
   The model forward pass is wrapped in a closure that:
     - pads token sequences to [B, MAX_CDR_LEN]
     - passes static-shaped arrays to Flax's ``model.apply``
     - extracts the last-position log-probs for beam expansion

5. **jax.jit on model forward**
   The model forward pass is wrapped with ``jax.jit`` for efficient
   repeated calls during beam search.  In PyTorch this is implicit via
   CUDA kernels; in JAX it requires an explicit ``jit`` call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from .model.transformer   import InverseFoldingTransformer, AA_VOCAB_SIZE
from .beam.search         import BeamSearchEngine, BeamResult, BOS_ID, MAX_CDR_LEN
from .scorer.delta_g      import DeltaGScorer
from .utils.structure     import (
    parse_pdb_chains,
    detect_cdr_h3,
    compute_epitope_adjacency,
    build_graph_tensors,
    residues_to_jax,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ForgeConfig:
    d_model:           int   = 256
    n_heads:           int   = 8
    n_layers:          int   = 6
    ffn_dim:           int   = 1024
    dropout_rate:      float = 0.1
    beam_width:        int   = 10_000
    max_cdr_len:       int   = MAX_CDR_LEN
    length_penalty:    float = 0.6
    top_k_results:     int   = 10
    n_slab_slots:      int   = 300_000
    max_candidates:    int   = 1_000
    n_rosetta_threads: int   = 8
    epitope_cutoff_A:  float = 6.0
    numbering_scheme:  str   = "kabat"
    seed:              int   = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ForgeConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class AntibodyForgePipeline:
    """
    End-to-end CDR-H3 inverse-folding pipeline — JAX/Flax version.
    """

    def __init__(
        self,
        config:          ForgeConfig,
        checkpoint_path: Optional[str | Path] = None,
    ) -> None:
        self.cfg    = config
        self._key   = jax.random.PRNGKey(config.seed)

        logger.info("Initialising InverseFoldingTransformer (JAX/Flax)")
        self.model = InverseFoldingTransformer(
            d_model      = config.d_model,
            n_heads      = config.n_heads,
            n_layers     = config.n_layers,
            ffn_dim      = config.ffn_dim,
            dropout_rate = config.dropout_rate,
        )

        # Initialise parameters with dummy inputs
        self.params = self._init_params()

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        logger.info("Initialising DeltaGScorer")
        self.scorer = DeltaGScorer(
            max_candidates    = config.max_candidates,
            n_rosetta_threads = config.n_rosetta_threads,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        **overrides,
    ) -> "AntibodyForgePipeline":
        cfg = ForgeConfig.from_yaml(config_path)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cls(cfg)

    # ------------------------------------------------------------------
    # Parameter initialisation
    # ------------------------------------------------------------------

    def _init_params(self) -> dict:
        """Initialise Flax parameters using dummy inputs."""
        cfg    = self.cfg
        dummy_tokens     = jnp.zeros(MAX_CDR_LEN, dtype=jnp.int32)
        dummy_ag_coords  = jnp.zeros((10, 3),  dtype=jnp.float32)
        dummy_ag_types   = jnp.zeros(10,        dtype=jnp.int32)
        dummy_ag_sasa    = jnp.zeros(10,        dtype=jnp.float32)
        dummy_row_ptr    = jnp.zeros(MAX_CDR_LEN + 1, dtype=jnp.int32)
        dummy_col_idx    = jnp.zeros(1,         dtype=jnp.int32)
        dummy_valid      = jnp.ones(MAX_CDR_LEN, dtype=bool)
        dummy_lengths    = jnp.array(7,         dtype=jnp.int32)

        self._key, init_key = jax.random.split(self._key)
        variables = self.model.init(
            init_key,
            dummy_tokens, dummy_ag_coords, dummy_ag_types, dummy_ag_sasa,
            dummy_row_ptr, dummy_col_idx, dummy_valid, dummy_lengths,
        )
        return variables["params"]

    # ------------------------------------------------------------------
    # JIT-compiled forward pass
    # ------------------------------------------------------------------

    def _make_jit_forward(
        self,
        ag_coords:  jnp.ndarray,
        ag_types:   jnp.ndarray,
        ag_sasa:    jnp.ndarray,
        row_ptr:    jnp.ndarray,
        col_idx:    jnp.ndarray,
        valid_mask: jnp.ndarray,
    ):
        """
        Return a JIT-compiled function:
            fn(token_ids [B, MAX_CDR_LEN], lengths [B]) → log_probs [B, VOCAB]

        Antigen tensors are closed over (static per protein pair).
        """
        params     = self.params
        model      = self.model

        @jax.jit
        def _forward(token_ids: jnp.ndarray, lengths: jnp.ndarray) -> jnp.ndarray:
            B = token_ids.shape[0]

            def single_forward(tok, length):
                lp = model.apply(
                    {"params": params},
                    tok, ag_coords, ag_types, ag_sasa,
                    row_ptr, col_idx, valid_mask, length,
                    deterministic=True,
                )  # [MAX_CDR_LEN, VOCAB]
                # Extract log-prob at position (length - 1) — last real token
                return lp[length - 1]    # [VOCAB]

            return jax.vmap(single_forward)(token_ids, lengths)  # [B, VOCAB]

        return _forward

    # ------------------------------------------------------------------
    # Main design entry point
    # ------------------------------------------------------------------

    def design(
        self,
        pdb_path:          str | Path,
        heavy_chain_id:    str       = "H",
        antigen_chain_ids: List[str] = None,
        top_k:             Optional[int] = None,
    ) -> List[BeamResult]:
        top_k    = top_k or self.cfg.top_k_results
        pdb_path = Path(pdb_path)
        logger.info("Parsing structure: %s", pdb_path)

        chains = parse_pdb_chains(pdb_path)
        heavy  = chains.get(heavy_chain_id, [])
        if not heavy:
            raise ValueError(f"Heavy chain '{heavy_chain_id}' not found in {pdb_path}")

        if antigen_chain_ids is None:
            antigen_chain_ids = [c for c in chains if c != heavy_chain_id]
        ag_residues = []
        for cid in antigen_chain_ids:
            ag_residues.extend(chains.get(cid, []))
        if not ag_residues:
            raise ValueError("No antigen residues found")

        cdr_residues = detect_cdr_h3(heavy, scheme=self.cfg.numbering_scheme)
        if not cdr_residues:
            raise RuntimeError("CDR-H3 detection failed")
        logger.info("CDR-H3 length: %d", len(cdr_residues))

        cdr_len, n_ag, edges = compute_epitope_adjacency(
            cdr_residues, ag_residues, cutoff=self.cfg.epitope_cutoff_A
        )
        # JAX: row_ptr, col_idx, valid_mask (static-shape padded)
        row_ptr, col_idx, valid_mask = build_graph_tensors(cdr_len, n_ag, edges)
        logger.info("Epitope edges (nnz): %d", len(edges))

        ag_coords, ag_types, ag_sasa = residues_to_jax(ag_residues)

        # JIT-compiled model forward closed over antigen context
        jit_forward = self._make_jit_forward(
            ag_coords, ag_types, ag_sasa, row_ptr, col_idx, valid_mask
        )

        logger.info("Starting beam search (width=%d)", self.cfg.beam_width)
        with BeamSearchEngine(
            model_logprob_fn = jit_forward,
            scorer           = self.scorer,
            beam_width       = self.cfg.beam_width,
            max_len          = min(self.cfg.max_cdr_len, len(cdr_residues) + 4),
            length_penalty   = self.cfg.length_penalty,
            top_k_results    = top_k,
            n_slots          = self.cfg.n_slab_slots,
        ) as engine:
            results = engine.search()

        logger.info("Design complete. Top: %s (ΔG=%.2f)", results[0].sequence, results[0].delta_g)
        return results

    # ------------------------------------------------------------------
    # Checkpoint I/O via Orbax
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: str | Path) -> None:
        """Load Flax params from an Orbax checkpoint directory."""
        try:
            import orbax.checkpoint as ocp
            path = Path(path)
            logger.info("Loading Orbax checkpoint: %s", path)
            checkpointer = ocp.PyTreeCheckpointer()
            self.params  = checkpointer.restore(str(path), item=self.params)
        except ImportError:
            logger.warning("orbax-checkpoint not installed; skipping checkpoint load")

    def save_checkpoint(self, path: str | Path) -> None:
        """Save Flax params to an Orbax checkpoint directory."""
        try:
            import orbax.checkpoint as ocp
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            checkpointer = ocp.PyTreeCheckpointer()
            checkpointer.save(str(path), self.params)
            logger.info("Orbax checkpoint saved: %s", path)
        except ImportError:
            logger.warning("orbax-checkpoint not installed; checkpoint not saved")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def results_to_fasta(results: List[BeamResult], path: str | Path) -> None:
        path = Path(path)
        with open(path, "w") as f:
            for i, r in enumerate(results, 1):
                f.write(f">design_{i:03d} logprob={r.logprob:.4f} dG={r.delta_g:.2f}\n")
                f.write(r.sequence + "\n")
        logger.info("FASTA written: %s (%d designs)", path, len(results))
