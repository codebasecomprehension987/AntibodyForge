"""
tests/unit/test_structure.py  (JAX)
=====================================
Unit tests for structure utilities — JAX version.
Main difference: residues_to_jax returns jnp arrays instead of torch.Tensor.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List

import jax.numpy as jnp
import numpy as np
import pytest

from src.utils.structure import (
    AA_ONE_TO_IDX,
    Residue,
    build_graph_tensors,
    compute_epitope_adjacency,
    detect_cdr_h3,
    parse_pdb_chains,
    residues_to_jax,
)
from src.kernels.sparse_cdr_attention import MAX_CDR_LEN


# ---------------------------------------------------------------------------
# Minimal PDB fixture
# ---------------------------------------------------------------------------

_MINI_PDB = textwrap.dedent("""\
ATOM      1  CA  ALA H  95       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  CYS H  96       2.000   3.000   4.000  1.00  0.00           C
ATOM      3  CA  ASP H  97       3.000   4.000   5.000  1.00  0.00           C
ATOM      4  CA  GLU H  98       4.000   5.000   6.000  1.00  0.00           C
ATOM      5  CA  PHE H  99       5.000   6.000   7.000  1.00  0.00           C
ATOM      6  CA  GLY H 100       6.000   7.000   8.000  1.00  0.00           C
ATOM      7  CA  HIS H 101       7.000   8.000   9.000  1.00  0.00           C
ATOM      8  CA  ILE A   1       1.500   2.500   3.500  1.00  0.00           C
ATOM      9  CA  LYS A   2      10.000  20.000  30.000  1.00  0.00           C
END
""")


@pytest.fixture
def mini_pdb(tmp_path: Path) -> Path:
    p = tmp_path / "mini.pdb"
    p.write_text(_MINI_PDB)
    return p


# ---------------------------------------------------------------------------
# Tests: parse_pdb_chains
# ---------------------------------------------------------------------------

class TestParsePdbChains:

    def test_chains_found(self, mini_pdb):
        chains = parse_pdb_chains(mini_pdb)
        assert set(chains.keys()) == {"H", "A"}

    def test_heavy_length(self, mini_pdb):
        assert len(parse_pdb_chains(mini_pdb)["H"]) == 7

    def test_residue_type(self, mini_pdb):
        for res in parse_pdb_chains(mini_pdb)["H"]:
            assert isinstance(res, Residue)

    def test_ca_xyz_shape(self, mini_pdb):
        for res in parse_pdb_chains(mini_pdb)["H"]:
            assert res.ca_xyz.shape == (3,)
            assert res.ca_xyz.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests: detect_cdr_h3
# ---------------------------------------------------------------------------

class TestDetectCdrH3:

    def _make_heavy(self, rng):
        return [
            Residue("H", i, "", "A",
                    np.array([float(i), 0.0, 0.0], dtype=np.float32), 0.0)
            for i in rng
        ]

    def test_kabat(self):
        cdr = detect_cdr_h3(self._make_heavy(range(80, 115)), scheme="kabat")
        assert all(95 <= r.res_seq <= 102 for r in cdr)

    def test_imgt(self):
        cdr = detect_cdr_h3(self._make_heavy(range(100, 120)), scheme="imgt")
        assert all(105 <= r.res_seq <= 117 for r in cdr)

    def test_empty_chain(self):
        assert detect_cdr_h3([], scheme="kabat") == []


# ---------------------------------------------------------------------------
# Tests: compute_epitope_adjacency
# ---------------------------------------------------------------------------

class TestComputeEpitopeAdjacency:

    def _res(self, xyzs):
        return [Residue("X", i, "", "A",
                        np.array(xyz, dtype=np.float32), 0.0)
                for i, xyz in enumerate(xyzs)]

    def test_close_pair(self):
        cdr = self._res([[0., 0., 0.]])
        ag  = self._res([[3., 0., 0.]])
        _, _, edges = compute_epitope_adjacency(cdr, ag, cutoff=6.0)
        assert (0, 0) in edges

    def test_distant_pair(self):
        cdr = self._res([[0., 0., 0.]])
        ag  = self._res([[100., 0., 0.]])
        _, _, edges = compute_epitope_adjacency(cdr, ag, cutoff=6.0)
        assert edges == []


# ---------------------------------------------------------------------------
# Tests: build_graph_tensors (JAX version returns jnp arrays)
# ---------------------------------------------------------------------------

class TestBuildGraphTensors:

    def test_shapes(self):
        rp, ci, vm = build_graph_tensors(3, 5, [(0, 1), (0, 3), (2, 4)])
        assert rp.shape == (MAX_CDR_LEN + 1,)
        assert ci.shape == (3,)
        assert vm.shape == (MAX_CDR_LEN,)

    def test_dtypes_jax(self):
        rp, ci, vm = build_graph_tensors(2, 4, [(0, 0)])
        assert rp.dtype == jnp.int32
        assert ci.dtype == jnp.int32
        assert vm.dtype == jnp.bool_

    def test_valid_mask_correct(self):
        rp, ci, vm = build_graph_tensors(7, 10, [(0, 0)])
        assert vm[:7].all()
        assert not vm[7:].any()


# ---------------------------------------------------------------------------
# Tests: residues_to_jax
# ---------------------------------------------------------------------------

class TestResiduesToJax:

    def _make(self, aas):
        return [
            Residue("A", i, "", aa,
                    np.array([float(i), 0., 0.], dtype=np.float32), 1.0)
            for i, aa in enumerate(aas)
        ]

    def test_shapes(self):
        residues = self._make("ACDEFGH")
        coords, aa_types, sasa = residues_to_jax(residues)
        assert coords.shape   == (7, 3)
        assert aa_types.shape == (7,)
        assert sasa.shape     == (7,)

    def test_dtypes(self):
        coords, aa_types, sasa = residues_to_jax(self._make("A"))
        assert coords.dtype   == jnp.float32
        assert aa_types.dtype == jnp.int32
        assert sasa.dtype     == jnp.float32

    def test_aa_index(self):
        _, aa_types, _ = residues_to_jax(self._make("A"))
        assert int(aa_types[0]) == AA_ONE_TO_IDX["A"]

    def test_jax_array_type(self):
        coords, aa_types, sasa = residues_to_jax(self._make("ACD"))
        assert isinstance(coords,   jnp.ndarray)
        assert isinstance(aa_types, jnp.ndarray)
        assert isinstance(sasa,     jnp.ndarray)
