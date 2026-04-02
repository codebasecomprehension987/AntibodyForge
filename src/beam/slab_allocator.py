"""
beam/slab_allocator.py
=======================
Identical ctypes bridge to the Rust slab allocator.
No PyTorch or JAX imports — pure Python / ctypes.
See the PyTorch version for full documentation.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

_LIB_NAME = "libbeam_slab.so"


def _find_lib() -> Path:
    search = [
        Path(__file__).parent / _LIB_NAME,
        Path(__file__).parent.parent.parent / "target" / "release" / _LIB_NAME,
    ]
    env_path = os.environ.get("BEAM_SLAB_LIB")
    if env_path:
        search.insert(0, Path(env_path))
    for p in search:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not locate {_LIB_NAME}. Run `cargo build --release` or set "
        "BEAM_SLAB_LIB=/path/to/libbeam_slab.so"
    )


class _SlabLib:
    def __init__(self, path: Path) -> None:
        self._lib = ctypes.CDLL(str(path))
        self._bind()

    def _bind(self) -> None:
        lib = self._lib
        lib.slab_create.restype  = ctypes.c_void_p
        lib.slab_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        lib.slab_alloc.restype   = ctypes.c_int64
        lib.slab_alloc.argtypes  = [ctypes.c_void_p]
        lib.slab_free.restype    = None
        lib.slab_free.argtypes   = [ctypes.c_void_p, ctypes.c_int64]
        lib.slab_write_seq.restype  = None
        lib.slab_write_seq.argtypes = [
            ctypes.c_void_p, ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32,
        ]
        lib.slab_read_seq.restype  = None
        lib.slab_read_seq.argtypes = [
            ctypes.c_void_p, ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.slab_write_logprob.restype  = None
        lib.slab_write_logprob.argtypes = [
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_float,
        ]
        lib.slab_read_logprob.restype  = ctypes.c_float
        lib.slab_read_logprob.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        lib.slab_write_parent.restype  = None
        lib.slab_write_parent.argtypes = [
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
        ]
        lib.slab_read_parent.restype  = ctypes.c_int64
        lib.slab_read_parent.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        lib.slab_destroy.restype  = None
        lib.slab_destroy.argtypes = [ctypes.c_void_p]
        lib.slab_stats.restype    = None
        lib.slab_stats.argtypes   = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]


class BeamSlabArena:
    """Rust slab arena bridge. Identical API to the PyTorch version."""

    _lib: Optional[_SlabLib] = None

    @classmethod
    def _get_lib(cls) -> _SlabLib:
        if cls._lib is None:
            cls._lib = _SlabLib(_find_lib())
        return cls._lib

    def __init__(self, n_slots: int = 12_000, max_cdr_len: int = 28) -> None:
        lib = self._get_lib()
        self._lib_ref   = lib
        self._max_cdr   = max_cdr_len
        self._arena_ptr = lib._lib.slab_create(
            ctypes.c_uint32(n_slots), ctypes.c_uint32(max_cdr_len)
        )
        if not self._arena_ptr:
            raise MemoryError("slab_create returned NULL")
        self._destroyed = False

    def alloc(self) -> int:
        self._check_alive()
        h = self._lib_ref._lib.slab_alloc(self._arena_ptr)
        if h < 0:
            raise MemoryError("Slab arena full")
        return int(h)

    def free(self, handle: int) -> None:
        self._check_alive()
        self._lib_ref._lib.slab_free(self._arena_ptr, ctypes.c_int64(handle))

    def write_seq(self, handle: int, token_ids: list[int]) -> None:
        self._check_alive()
        arr = (ctypes.c_int32 * len(token_ids))(*token_ids)
        self._lib_ref._lib.slab_write_seq(
            self._arena_ptr, ctypes.c_int64(handle), arr,
            ctypes.c_uint32(len(token_ids)),
        )

    def read_seq(self, handle: int) -> list[int]:
        self._check_alive()
        buf    = (ctypes.c_int32 * self._max_cdr)()
        length = ctypes.c_uint32(0)
        self._lib_ref._lib.slab_read_seq(
            self._arena_ptr, ctypes.c_int64(handle),
            buf, ctypes.byref(length),
        )
        return list(buf[: length.value])

    def write_logprob(self, handle: int, lp: float) -> None:
        self._check_alive()
        self._lib_ref._lib.slab_write_logprob(
            self._arena_ptr, ctypes.c_int64(handle), ctypes.c_float(lp)
        )

    def read_logprob(self, handle: int) -> float:
        self._check_alive()
        return float(self._lib_ref._lib.slab_read_logprob(
            self._arena_ptr, ctypes.c_int64(handle)
        ))

    def write_parent(self, handle: int, parent: int) -> None:
        self._check_alive()
        self._lib_ref._lib.slab_write_parent(
            self._arena_ptr, ctypes.c_int64(handle), ctypes.c_int64(parent)
        )

    def read_parent(self, handle: int) -> int:
        self._check_alive()
        return int(self._lib_ref._lib.slab_read_parent(
            self._arena_ptr, ctypes.c_int64(handle)
        ))

    def stats(self) -> dict[str, int]:
        self._check_alive()
        n_used = ctypes.c_uint32(0)
        n_free = ctypes.c_uint32(0)
        self._lib_ref._lib.slab_stats(
            self._arena_ptr, ctypes.byref(n_used), ctypes.byref(n_free)
        )
        return {"n_used": n_used.value, "n_free": n_free.value}

    def destroy(self) -> None:
        if not self._destroyed:
            self._lib_ref._lib.slab_destroy(self._arena_ptr)
            self._destroyed = True

    def __del__(self)              -> None: self.destroy()
    def __enter__(self)            -> "BeamSlabArena": return self
    def __exit__(self, *_)         -> None: self.destroy()
    def _check_alive(self)         -> None:
        if self._destroyed:
            raise RuntimeError("BeamSlabArena has been destroyed")
