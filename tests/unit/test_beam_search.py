"""
tests/unit/test_beam_search.py  (JAX)
======================================
Unit tests for the JAX beam search engine.
Uses Python slab stub (no Rust required) and JAX mock model.
"""

from __future__ import annotations

import math
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Python slab stub
# ---------------------------------------------------------------------------

class _PythonSlabArena:
    def __init__(self, n_slots=50_000, max_cdr_len=28):
        self._slots = {}
        self._free  = list(range(n_slots - 1, -1, -1))

    def alloc(self):
        h = self._free.pop()
        self._slots[h] = {"seq": [], "logprob": 0.0, "parent": -1}
        return h

    def free(self, h):
        if h in self._slots:
            del self._slots[h]
            self._free.append(h)

    def write_seq(self, h, seq):      self._slots[h]["seq"] = list(seq)
    def read_seq(self, h):            return list(self._slots[h]["seq"])
    def write_logprob(self, h, lp):   self._slots[h]["logprob"] = lp
    def read_logprob(self, h):        return self._slots[h]["logprob"]
    def write_parent(self, h, p):     self._slots[h]["parent"] = p
    def read_parent(self, h):         return self._slots[h]["parent"]
    def stats(self):                  return {"n_used": len(self._slots), "n_free": len(self._free)}
    def destroy(self):                pass
    def __enter__(self):              return self
    def __exit__(self, *_):           pass


# ---------------------------------------------------------------------------
# Mock model functions (JAX signature)
# ---------------------------------------------------------------------------

VOCAB = 23

def _uniform_model(token_ids: jnp.ndarray, lengths: jnp.ndarray) -> jnp.ndarray:
    """Returns uniform log-probs. token_ids: [B, MAX_LEN], lengths: [B]"""
    B = token_ids.shape[0]
    return jnp.full((B, VOCAB), -math.log(VOCAB))


def _greedy_model(token_ids: jnp.ndarray, lengths: jnp.ndarray) -> jnp.ndarray:
    """Always predicts token 3 until length 5, then EOS."""
    B = token_ids.shape[0]
    lp = jnp.full((B, VOCAB), -jnp.inf)

    def one(b):
        t = lengths[b]
        return jnp.where(t >= 5,
                         lp.at[b, 2].set(0.0),   # EOS
                         lp.at[b, 3].set(0.0))    # AA token

    for b in range(B):
        lp = one(b)
    return lp


# ---------------------------------------------------------------------------
# Mock scorer
# ---------------------------------------------------------------------------

class _MockScorer:
    def __init__(self, max_candidates=1_000):
        self.score_buf = np.zeros(max_candidates, dtype=np.float32)

    def score_batch(self, seqs, out_buf):
        out_buf[:len(seqs)] = 0.0


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def _make_engine(model_fn=None, beam_width=4, max_len=6):
    from src.beam.search import BeamSearchEngine
    scorer = _MockScorer()
    engine = BeamSearchEngine.__new__(BeamSearchEngine)
    engine.model_fn   = model_fn or _uniform_model
    engine.scorer     = scorer
    engine.beam_width = beam_width
    engine.max_len    = max_len
    engine.lp_alpha   = 0.6
    engine.top_k      = 2
    engine._score_buf = scorer.score_buf
    engine._arena     = _PythonSlabArena(n_slots=100_000, max_cdr_len=max_len)
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBeamSearchEngineJax:

    def test_returns_correct_count(self):
        engine  = _make_engine(beam_width=8, max_len=5)
        results = engine.search()
        assert len(results) == engine.top_k

    def test_results_are_beam_results(self):
        from src.beam.search import BeamResult
        results = _make_engine().search()
        for r in results:
            assert isinstance(r, BeamResult)

    def test_sorted_by_logprob(self):
        results = _make_engine(beam_width=16, max_len=7).search()
        lps = [r.logprob for r in results]
        assert lps == sorted(lps, reverse=True)

    def test_score_buf_is_numpy_float32(self):
        engine = _make_engine()
        assert isinstance(engine._score_buf, np.ndarray)
        assert engine._score_buf.dtype == np.float32

    def test_arena_empty_after_search(self):
        engine = _make_engine(beam_width=8, max_len=4)
        engine.search()
        assert engine._arena.stats()["n_used"] == 0

    def test_beam_width_one(self):
        engine       = _make_engine(beam_width=1, max_len=5)
        engine.top_k = 1
        results      = engine.search()
        assert len(results) == 1

    def test_jax_top_k_used(self):
        """Verify jax.lax.top_k is callable (JAX-specific test)."""
        scores = jnp.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.6])
        vals, idx = jax.lax.top_k(scores, 3)
        assert int(idx[0]) == 4   # 9.0 is the max
        assert float(vals[0]) == 9.0
