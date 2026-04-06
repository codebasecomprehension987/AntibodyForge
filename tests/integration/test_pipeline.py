"""
tests/integration/test_pipeline.py  (JAX)
==========================================
End-to-end integration test for the JAX AntibodyForge pipeline.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic PDB
# ---------------------------------------------------------------------------

def _write_synthetic_pdb(path: Path) -> None:
    heavy_aas = list("ACDEFGHIKLMNPQRS")
    three_map = {
        "A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE",
        "G":"GLY","H":"HIS","I":"ILE","K":"LYS","L":"LEU",
        "M":"MET","N":"ASN","P":"PRO","Q":"GLN","R":"ARG",
        "S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR",
    }
    lines = []
    for i, aa in enumerate(heavy_aas):
        rs  = 90 + i
        x, y, z = float(i), float(i * 0.5), 0.0
        lines.append(
            f"ATOM  {i+1:5d}  CA  {three_map[aa]} H{rs:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    ag_map = {"M":"MET","N":"ASN","P":"PRO","Q":"GLN","R":"ARG"}
    for j, aa in enumerate("MNPQR"):
        x, y, z = float(j + 5) + 0.1, float(j * 0.5) + 0.1, 3.5
        serial  = len(heavy_aas) + j + 1
        lines.append(
            f"ATOM  {serial:5d}  CA  {ag_map[aa]} A{j+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


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

    def write_seq(self, h, s):    self._slots[h]["seq"]     = list(s)
    def read_seq(self, h):        return list(self._slots[h]["seq"])
    def write_logprob(self, h, l): self._slots[h]["logprob"] = l
    def read_logprob(self, h):    return self._slots[h]["logprob"]
    def write_parent(self, h, p): self._slots[h]["parent"]  = p
    def read_parent(self, h):     return self._slots[h]["parent"]
    def stats(self):              return {"n_used": len(self._slots), "n_free": len(self._free)}
    def destroy(self):            pass
    def __enter__(self):          return self
    def __exit__(self, *_):       pass


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def pdb_path(tmp_path) -> Path:
    p = tmp_path / "synthetic.pdb"
    _write_synthetic_pdb(p)
    return p


@pytest.fixture
def config_path(tmp_path) -> Path:
    cfg = textwrap.dedent("""\
        d_model: 32
        n_heads: 2
        n_layers: 1
        ffn_dim: 64
        dropout_rate: 0.0
        beam_width: 4
        max_cdr_len: 10
        length_penalty: 0.6
        top_k_results: 2
        n_slab_slots: 5000
        max_candidates: 50
        n_rosetta_threads: 1
        epitope_cutoff_A: 8.0
        numbering_scheme: kabat
        seed: 0
    """)
    p = tmp_path / "test_config.yaml"
    p.write_text(cfg)
    return p


def _patch_slab():
    return patch(
        "src.beam.search.BeamSlabArena",
        side_effect=lambda **kw: _PythonSlabArena(**kw),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineJax:

    def test_design_returns_beam_results(self, pdb_path, config_path):
        from src.beam.search import BeamResult
        from src.pipeline import AntibodyForgePipeline

        with _patch_slab():
            pipeline = AntibodyForgePipeline.from_config(config_path)
            results  = pipeline.design(
                pdb_path          = pdb_path,
                heavy_chain_id    = "H",
                antigen_chain_ids = ["A"],
                top_k             = 2,
            )

        assert len(results) == 2
        for r in results:
            assert isinstance(r, BeamResult)

    def test_sequences_valid_amino_acids(self, pdb_path, config_path):
        from src.pipeline import AntibodyForgePipeline
        valid = set("ACDEFGHIKLMNPQRSTVWY")

        with _patch_slab():
            pipeline = AntibodyForgePipeline.from_config(config_path)
            results  = pipeline.design(pdb_path=pdb_path,
                                       heavy_chain_id="H",
                                       antigen_chain_ids=["A"])

        for r in results:
            for aa in r.sequence:
                assert aa in valid

    def test_sorted_best_first(self, pdb_path, config_path):
        from src.pipeline import AntibodyForgePipeline

        with _patch_slab():
            pipeline = AntibodyForgePipeline.from_config(config_path)
            results  = pipeline.design(pdb_path=pdb_path,
                                       heavy_chain_id="H",
                                       antigen_chain_ids=["A"])

        lps = [r.logprob for r in results]
        assert lps == sorted(lps, reverse=True)

    def test_fasta_export(self, pdb_path, config_path, tmp_path):
        from src.pipeline import AntibodyForgePipeline

        out = tmp_path / "out.fasta"
        with _patch_slab():
            pipeline = AntibodyForgePipeline.from_config(config_path)
            results  = pipeline.design(pdb_path=pdb_path,
                                       heavy_chain_id="H",
                                       antigen_chain_ids=["A"])
        AntibodyForgePipeline.results_to_fasta(results, out)

        assert out.exists()
        assert out.read_text().count(">design_") == len(results)

    def test_params_are_jax_pytree(self, config_path):
        """Flax params must be a valid JAX pytree (not torch tensors)."""
        from src.pipeline import AntibodyForgePipeline

        pipeline = AntibodyForgePipeline.from_config(config_path)
        leaves   = jax.tree_util.tree_leaves(pipeline.params)
        assert len(leaves) > 0
        for leaf in leaves:
            assert isinstance(leaf, jnp.ndarray), (
                f"Expected jnp.ndarray, got {type(leaf)}"
            )

    def test_missing_heavy_chain_raises(self, pdb_path, config_path):
        from src.pipeline import AntibodyForgePipeline

        with _patch_slab():
            pipeline = AntibodyForgePipeline.from_config(config_path)
            with pytest.raises(ValueError, match="Heavy chain"):
                pipeline.design(pdb_path=pdb_path, heavy_chain_id="Z")

    def test_jit_forward_compiles(self, pdb_path, config_path):
        """JIT compilation of the model forward pass must succeed."""
        from src.pipeline import AntibodyForgePipeline
        from src.utils.structure import (
            parse_pdb_chains, detect_cdr_h3, compute_epitope_adjacency,
            build_graph_tensors, residues_to_jax,
        )

        pipeline = AntibodyForgePipeline.from_config(config_path)
        chains   = parse_pdb_chains(pdb_path)
        heavy    = chains["H"]
        ag_res   = chains["A"]
        cdr_res  = detect_cdr_h3(heavy)
        cl, na, edges = compute_epitope_adjacency(cdr_res, ag_res, cutoff=8.0)
        rp, ci, vm = build_graph_tensors(cl, na, edges)
        ag_c, ag_t, ag_s = residues_to_jax(ag_res)

        jit_fwd = pipeline._make_jit_forward(ag_c, ag_t, ag_s, rp, ci, vm)

        from src.kernels.sparse_cdr_attention import MAX_CDR_LEN
        tokens  = jnp.zeros((2, MAX_CDR_LEN), dtype=jnp.int32)
        lengths = jnp.array([7, 7], dtype=jnp.int32)
        out     = jit_fwd(tokens, lengths)
        assert out.shape[0] == 2
