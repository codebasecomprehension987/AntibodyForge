"""
beam/search.py  (JAX)
======================
Beam search engine for CDR-H3 sequences — JAX rewrite.

JAX-specific changes vs the PyTorch version
--------------------------------------------

1. **jax.lax.top_k instead of torch.topk**
   ``jax.lax.top_k(flat_scores, k)`` returns (values, indices) exactly like
   ``torch.topk`` but operates on JAX arrays.

2. **No torch.no_grad() context**
   JAX is functional by default — no gradient tape is active unless you
   explicitly call ``jax.grad`` or ``jax.value_and_grad``.  The model forward
   pass inside the beam loop is naturally gradient-free.

3. **jnp arrays for score tensors**
   All tensor operations (log-prob combination, top-k selection) use
   ``jnp`` instead of ``torch``.  The numpy score buffer (GC-free trick)
   is unchanged — it is a plain numpy array, not a JAX array, so the CFFI
   scorer can write a C pointer into it without touching JAX's dispatch.

4. **Static shapes**
   JAX requires static shapes for JIT.  Partial sequences passed to the
   model are padded to MAX_CDR_LEN with PAD_ID=0 and a length integer is
   passed separately.  The model uses a causal mask derived from the length.

5. **Random key management**
   JAX uses explicit PRNG keys.  The beam search does not need randomness
   (it is deterministic top-k), but if stochastic sampling is added later,
   keys must be threaded explicitly through the beam state.

Slab allocator
--------------
Unchanged — still the Rust slab with Python ctypes bridge.  JAX arrays are
never stored in the slab; only integer token IDs and float32 log-probs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .slab_allocator import BeamSlabArena

PAD_ID        = 0
BOS_ID        = 1
EOS_ID        = 2
AA_VOCAB_SIZE = 23
MAX_CDR_LEN   = 28
MIN_CDR_LEN   = 7


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BeamResult:
    """Final decoded CDR-H3 sequence with associated scores."""
    token_ids: List[int]
    logprob:   float
    delta_g:   float
    sequence:  str

    _AA_TOKENS = list("ACDEFGHIKLMNPQRSTVWY")

    def to_fasta(self, header: str = "CDR-H3") -> str:
        return f">{header}\n{self.sequence}\n"


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

class BeamSearchEngine:
    """
    JAX beam search for CDR-H3 inverse folding.

    Model function signature
    ------------------------
    ``model_logprob_fn(token_ids, lengths) -> log_probs``

      token_ids : jnp.ndarray [B, MAX_CDR_LEN]  int32 — padded partial seqs
      lengths   : jnp.ndarray [B]                int32 — actual seq lengths
      returns   : jnp.ndarray [B, VOCAB_SIZE]    float32 — next-token log-probs

    This differs from the PyTorch version (which passed variable-length
    sequences) to satisfy JAX's static-shape requirement.
    """

    _AA_TOKENS = list("ACDEFGHIKLMNPQRSTVWY")

    def __init__(
        self,
        model_logprob_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        scorer,
        beam_width:     int   = 10_000,
        max_len:        int   = MAX_CDR_LEN,
        length_penalty: float = 0.6,
        top_k_results:  int   = 10,
        n_slots:        int   = 300_000,
    ) -> None:
        self.model_fn      = model_logprob_fn
        self.scorer        = scorer
        self.beam_width    = beam_width
        self.max_len       = max_len
        self.lp_alpha      = length_penalty
        self.top_k         = top_k_results

        # Pre-allocated GC-free score buffer (numpy, not JAX array)
        self._score_buf: np.ndarray = np.empty(beam_width, dtype=np.float32)

        self._arena = BeamSlabArena(n_slots=n_slots, max_cdr_len=max_len)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def search(self) -> List[BeamResult]:
        """Run beam search. Returns top-k BeamResult objects."""
        arena = self._arena

        # Seed beam
        active_handles: List[int] = []
        for _ in range(self.beam_width):
            h = arena.alloc()
            arena.write_seq(h, [BOS_ID])
            arena.write_logprob(h, 0.0)
            arena.write_parent(h, -1)
            active_handles.append(h)

        finished_handles: List[int] = []

        for step in range(self.max_len):
            if not active_handles:
                break

            B = len(active_handles)

            # Build padded token tensor + lengths for JAX model
            # Shape: [B, MAX_CDR_LEN]  — JAX static shape requirement
            token_arr = np.zeros((B, self.max_len), dtype=np.int32)
            lengths   = np.zeros(B, dtype=np.int32)
            for i, h in enumerate(active_handles):
                seq = arena.read_seq(h)
                L   = min(len(seq), self.max_len)
                token_arr[i, :L] = seq[:L]
                lengths[i]       = L

            # JAX arrays
            jax_tokens  = jnp.array(token_arr)   # [B, max_len]
            jax_lengths = jnp.array(lengths)      # [B]

            # Model forward — no grad context needed in JAX
            log_probs = self.model_fn(jax_tokens, jax_lengths)  # [B, VOCAB]

            # Parent log-probs as JAX array
            parent_lps = jnp.array(
                [arena.read_logprob(h) for h in active_handles],
                dtype=jnp.float32,
            )  # [B]

            # Combined scores: [B, VOCAB]
            combined = parent_lps[:, None] + log_probs

            # Flatten → [B×VOCAB] then top-k via jax.lax.top_k
            combined_flat = combined.reshape(-1)
            n_candidates  = min(2 * self.beam_width, combined_flat.shape[0])

            # jax.lax.top_k returns (values, indices) — JAX equivalent of torch.topk
            top_vals, top_flat_idx = jax.lax.top_k(combined_flat, n_candidates)

            # Move to numpy for Python-level beam bookkeeping
            top_flat_idx_np = np.array(top_flat_idx)
            top_vals_np     = np.array(top_vals)

            parent_idx = (top_flat_idx_np // AA_VOCAB_SIZE).tolist()
            token_idx  = (top_flat_idx_np %  AA_VOCAB_SIZE).tolist()
            cand_lp    = top_vals_np.tolist()

            # Build candidate sequences
            cand_seqs = []
            for k in range(n_candidates):
                parent_seq = arena.read_seq(active_handles[parent_idx[k]])
                cand_seqs.append(parent_seq + [token_idx[k]])

            # GC-free ΔG scoring — numpy buffer, no Python float objects
            score_view = self._score_buf[:n_candidates]
            self.scorer.score_batch(cand_seqs, score_view)

            # Combine objective with length penalty
            t        = step + 1
            lp_denom = ((5 + t) / 6) ** self.lp_alpha
            obj      = np.array(cand_lp, dtype=np.float32) - score_view
            obj     /= lp_denom

            # Select top beam_width survivors
            best_k   = min(self.beam_width, n_candidates)
            best_idx = np.argpartition(-obj, best_k - 1)[:best_k]
            best_idx = best_idx[np.argsort(-obj[best_idx])]

            new_handles: List[int] = []
            for ci in best_idx:
                pid = parent_idx[ci]
                tok = token_idx[ci]
                lp  = cand_lp[ci]
                seq = cand_seqs[ci]

                new_h = arena.alloc()
                arena.write_seq(new_h, seq)
                arena.write_logprob(new_h, lp)
                arena.write_parent(new_h, active_handles[pid])

                if tok == EOS_ID or len(seq) - 1 >= self.max_len:
                    finished_handles.append(new_h)
                else:
                    new_handles.append(new_h)

            surviving_parents = {parent_idx[ci] for ci in best_idx}
            for i, h in enumerate(active_handles):
                if i not in surviving_parents:
                    arena.free(h)

            active_handles = new_handles

        finished_handles.extend(active_handles)
        results = self._decode_top_k(finished_handles)
        for h in finished_handles:
            arena.free(h)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _decode_top_k(self, handles: List[int]) -> List[BeamResult]:
        arena  = self._arena
        scored = []
        for h in handles:
            seq    = arena.read_seq(h)
            lp     = arena.read_logprob(h)
            aa_ids = [t - 3 for t in seq if 3 <= t < 23]
            aa_str = "".join(
                self._AA_TOKENS[i] for i in aa_ids if 0 <= i < 20
            )
            dg = float(self._score_buf[0])
            scored.append((lp, dg, seq, aa_str))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            BeamResult(token_ids=seq, logprob=lp, delta_g=dg, sequence=aa)
            for lp, dg, seq, aa in scored[: self.top_k]
        ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __del__(self)        -> None: self._arena.destroy()
    def __enter__(self)      -> "BeamSearchEngine": return self
    def __exit__(self, *_)   -> None: self._arena.destroy()
