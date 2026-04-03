"""
scorer/delta_g.py  (JAX)
=========================
GC-free surrogate ΔG scorer — identical to the PyTorch version.

This module has ZERO JAX or PyTorch imports.  The pre-allocated numpy
buffer trick and CFFI Rosetta shim work identically regardless of the
deep-learning framework.  This is the biggest portability win in the
entire codebase: the GC-pressure solution is framework-agnostic.

See the PyTorch version for full documentation.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

_SHIM_NAME = "librosetta_shim.so"


def _find_shim() -> Optional[Path]:
    search = [
        Path(__file__).parent / _SHIM_NAME,
        Path(os.environ.get("ROSETTA_SHIM_LIB", "")),
    ]
    for p in search:
        if p and p.exists():
            return p
    return None


class _RosettaShim:
    def __init__(self, n_threads: int = 8) -> None:
        self.n_threads  = n_threads
        self._lib       = None
        self._mock_mode = False

        shim_path = _find_shim()
        if shim_path is None:
            self._mock_mode = True
            return

        lib = ctypes.CDLL(str(shim_path))
        lib.rosetta_fastrelax_batch.restype  = None
        lib.rosetta_fastrelax_batch.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self._lib = lib

    def call(
        self,
        seqs:      List[List[int]],
        out_buf:   np.ndarray,
        aa_tokens: List[str],
    ) -> None:
        n = len(seqs)
        if self._mock_mode or self._lib is None:
            rng = np.random.default_rng(seed=42)
            out_buf[:n] = rng.uniform(-20.0, 0.0, size=n).astype(np.float32)
            return

        def _tok_to_str(tok_ids: List[int]) -> bytes:
            return "".join(
                aa_tokens[t - 3] for t in tok_ids
                if 3 <= t < 3 + len(aa_tokens)
            ).encode("ascii")

        c_strs   = [ctypes.c_char_p(_tok_to_str(s)) for s in seqs]
        arr_type = ctypes.c_char_p * n
        c_arr    = arr_type(*c_strs)
        out_ptr  = out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._lib.rosetta_fastrelax_batch(c_arr, n, out_ptr, self.n_threads)


_AA_TOKENS = list("ACDEFGHIKLMNPQRSTVWY")


class DeltaGScorer:
    """
    GC-free ΔG scorer — framework-agnostic (no JAX/PyTorch imports).

    Pre-allocates a single numpy float32 buffer.  The CFFI scorer writes
    directly into it via a C pointer.  No Python float objects are created
    inside the beam loop.
    """

    def __init__(
        self,
        max_candidates:    int       = 1_000,
        n_rosetta_threads: int       = 8,
        aa_tokens:         List[str] = _AA_TOKENS,
    ) -> None:
        # Single pre-allocated buffer — reused for every beam step
        self.score_buf: np.ndarray = np.empty(max_candidates, dtype=np.float32)
        self._max   = max_candidates
        self._aa    = aa_tokens
        self._shim  = _RosettaShim(n_threads=n_rosetta_threads)

    def score_batch(self, seqs: List[List[int]], out_buf: np.ndarray) -> None:
        """Write ΔG values into out_buf in-place. No Python floats created."""
        n = len(seqs)
        if n == 0:
            return
        if n > self._max:
            raise ValueError(
                f"score_batch: {n} sequences > max_candidates={self._max}"
            )
        self._shim.call(seqs, out_buf, self._aa)

    def score_single(self, seq: List[int]) -> float:
        """Score one sequence. Returns Python float (for reporting only)."""
        view = self.score_buf[:1]
        self.score_batch([seq], view)
        return float(view[0])
