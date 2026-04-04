"""Utility sub-package — JAX version."""

from .structure import (
    parse_pdb_chains,
    detect_cdr_h3,
    compute_epitope_adjacency,
    build_graph_tensors,
    residues_to_jax,
    Residue,
    AA_THREE_TO_ONE,
    AA_ONE_TO_IDX,
)

__all__ = [
    "parse_pdb_chains",
    "detect_cdr_h3",
    "compute_epitope_adjacency",
    "build_graph_tensors",
    "residues_to_jax",
    "Residue",
    "AA_THREE_TO_ONE",
    "AA_ONE_TO_IDX",
]
