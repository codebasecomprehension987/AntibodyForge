"""
tests/unit/test_scorer.py  (JAX)
=================================
Unit tests for the GC-free DeltaGScorer — identical assertions to the
PyTorch version because the scorer has zero framework dependency.
"""

from __future__ import annotations

import gc
import tracemalloc
from typing import List

import numpy as np
import pytest

from src.scorer.delta_g import DeltaGScorer


def _make_scorer(max_candidates: int = 100) -> DeltaGScorer:
    return DeltaGScorer(max_candidates=max_candidates, n_rosetta_threads=1)


def _random_seqs(n: int, length: int = 8) -> List[List[int]]:
    rng = np.random.default_rng(seed=7)
    return [rng.integers(3, 23, size=length).tolist() for _ in range(n)]


class TestDeltaGScorerBuffer:

    def test_score_buf_is_numpy_float32(self):
        scorer = _make_scorer()
        assert isinstance(scorer.score_buf, np.ndarray)
        assert scorer.score_buf.dtype == np.float32

    def test_score_buf_length(self):
        scorer = _make_scorer(max_candidates=512)
        assert len(scorer.score_buf) == 512

    def test_same_object_across_calls(self):
        scorer = _make_scorer()
        id0 = id(scorer.score_buf)
        view = scorer.score_buf[:5]
        scorer.score_batch(_random_seqs(5), view)
        scorer.score_batch(_random_seqs(5), view)
        assert id(scorer.score_buf) == id0


class TestScoreBatch:

    def test_writes_values(self):
        scorer = _make_scorer(max_candidates=50)
        view   = scorer.score_buf[:10]
        scorer.score_batch(_random_seqs(10), view)
        assert view.shape    == (10,)
        assert view.dtype    == np.float32

    def test_empty_batch_ok(self):
        scorer = _make_scorer()
        scorer.score_batch([], scorer.score_buf[:0])

    def test_raises_oversized(self):
        scorer = _make_scorer(max_candidates=10)
        with pytest.raises(ValueError, match="max_candidates"):
            scorer.score_batch(_random_seqs(11), scorer.score_buf[:11])

    def test_mock_values_plausible(self):
        scorer = _make_scorer(max_candidates=100)
        view   = scorer.score_buf[:50]
        scorer.score_batch(_random_seqs(50), view)
        assert np.all(view[:50] >= -25.0)
        assert np.all(view[:50] <=   5.0)

    def test_no_python_float_objects_created(self):
        scorer = _make_scorer(max_candidates=1_000)
        seqs   = _random_seqs(1_000)
        view   = scorer.score_buf[:1_000]

        gc.collect()
        tracemalloc.start()
        scorer.score_batch(seqs, view)
        snap = tracemalloc.take_snapshot()
        tracemalloc.stop()

        float_allocs = sum(
            s.count for s in snap.statistics("lineno")
            if "float" in str(s).lower()
        )
        assert float_allocs < 50, (
            f"score_batch created {float_allocs} Python float objects"
        )


class TestScoreSingle:

    def test_returns_python_float(self):
        scorer = _make_scorer()
        assert isinstance(scorer.score_single([3, 4, 5, 6]), float)

    def test_consistent_with_batch(self):
        scorer = _make_scorer()
        seq    = [3, 7, 12, 5, 9]
        view   = scorer.score_buf[:1]
        scorer.score_batch([seq], view)
        batch_val = float(view[0])

        scorer2    = _make_scorer()
        single_val = scorer2.score_single(seq)
        assert abs(batch_val - single_val) < 1e-5
