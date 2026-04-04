"""
utils/structure.py  (JAX)
==========================
Structure parsing and epitope adjacency utilities.

Identical to the PyTorch version except:
  - ``residues_to_tensors`` returns JAX arrays (jnp) instead of torch.Tensor
  - ``build_graph_tensors`` returns JAX arrays and includes the valid_mask
    required by the JAX static-shape model
  - No torch imports anywhere

The core PDB parsing, CDR-H3 detection, and distance-matrix computation
are pure numpy — framework-agnostic.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
import jax.numpy as jnp

from ..kernels.sparse_cdr_attention import build_csr_adjacency, MAX_CDR_LEN


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

AA_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}
AA_ONE_TO_IDX: Dict[str, int] = {
    aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")
}


class Residue(NamedTuple):
    chain_id:  str
    res_seq:   int
    ins_code:  str
    aa_one:    str
    ca_xyz:    np.ndarray   # [3] float32
    sasa:      float


# ---------------------------------------------------------------------------
# PDB parser
# ---------------------------------------------------------------------------

_ATOM_RE = re.compile(
    r"^ATOM\s{2}(?P<serial>\d{5})\s(?P<name>.{4})\s?(?P<resname>.{3})\s"
    r"(?P<chain>.)(?P<resseq>.{4})(?P<icode>.)\s{3}"
    r"(?P<x>.{8})(?P<y>.{8})(?P<z>.{8})",
    re.ASCII,
)


def _iter_ca_records(pdb_path: Path) -> Iterator[Residue]:
    with open(pdb_path, "r", encoding="ascii", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            m = _ATOM_RE.match(line)
            if m is None or m.group("name").strip() != "CA":
                continue
            resname = m.group("resname").strip()
            aa_one  = AA_THREE_TO_ONE.get(resname, "X")
            xyz     = np.array([float(m.group("x")), float(m.group("y")),
                                float(m.group("z"))], dtype=np.float32)
            yield Residue(
                chain_id=m.group("chain"),
                res_seq=int(m.group("resseq")),
                ins_code=m.group("icode").strip(),
                aa_one=aa_one,
                ca_xyz=xyz,
                sasa=0.0,
            )


def parse_pdb_chains(pdb_path: str | Path) -> Dict[str, List[Residue]]:
    chains: Dict[str, List[Residue]] = {}
    for res in _iter_ca_records(Path(pdb_path)):
        if res.aa_one == "X":
            continue
        chains.setdefault(res.chain_id, []).append(res)
    return chains


# ---------------------------------------------------------------------------
# CDR-H3 detection
# ---------------------------------------------------------------------------

_CDR_H3_RANGES = {
    "kabat":   (95,  102),
    "imgt":    (105, 117),
    "chothia": (95,  102),
}


def detect_cdr_h3(
    heavy_chain: List[Residue],
    scheme: str = "kabat",
) -> List[Residue]:
    lo, hi = _CDR_H3_RANGES.get(scheme, _CDR_H3_RANGES["kabat"])
    cdr = [r for r in heavy_chain if lo <= r.res_seq <= hi]
    if not cdr:
        cdr = _detect_by_loop_length(heavy_chain)
    return cdr


def _detect_by_loop_length(heavy_chain: List[Residue]) -> List[Residue]:
    best: List[Residue] = []
    for i, r in enumerate(heavy_chain):
        if r.aa_one != "C":
            continue
        for j in range(i + 7, min(i + 29, len(heavy_chain))):
            if heavy_chain[j].aa_one == "W":
                loop = heavy_chain[i + 1: j]
                if len(loop) > len(best):
                    best = loop
    return best


# ---------------------------------------------------------------------------
# Epitope adjacency
# ---------------------------------------------------------------------------

def compute_epitope_adjacency(
    cdr_residues: List[Residue],
    ag_residues:  List[Residue],
    cutoff: float = 6.0,
) -> Tuple[int, int, List[Tuple[int, int]]]:
    cdr_xyz = np.stack([r.ca_xyz for r in cdr_residues])
    ag_xyz  = np.stack([r.ca_xyz for r in ag_residues])
    diff    = cdr_xyz[:, None, :] - ag_xyz[None, :, :]
    dist2   = (diff ** 2).sum(axis=-1)
    cdr_i, ag_j = np.where(dist2 <= cutoff ** 2)
    edges = list(zip(cdr_i.tolist(), ag_j.tolist()))
    return len(cdr_residues), len(ag_residues), edges


def build_graph_tensors(
    cdr_len:   int,
    n_ag:      int,
    edge_list: List[Tuple[int, int]],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build JAX CSR tensors from an epitope edge list.

    Returns
    -------
    row_ptr    : jnp.ndarray int32 [MAX_CDR_LEN + 1]
    col_idx    : jnp.ndarray int32 [nnz]
    valid_mask : jnp.ndarray bool  [MAX_CDR_LEN]
    """
    row_ptr_np, col_idx_np, valid_np = build_csr_adjacency(
        cdr_len, n_ag, edge_list
    )
    return (
        jnp.array(row_ptr_np, dtype=jnp.int32),
        jnp.array(col_idx_np, dtype=jnp.int32),
        jnp.array(valid_np,   dtype=bool),
    )


# ---------------------------------------------------------------------------
# Tensor builders
# ---------------------------------------------------------------------------

def residues_to_jax(
    residues: List[Residue],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convert Residue list → JAX arrays.

    Returns
    -------
    coords   : float32 [N, 3]
    aa_types : int32   [N]
    sasa     : float32 [N]
    """
    coords   = jnp.array(
        np.stack([r.ca_xyz for r in residues]), dtype=jnp.float32
    )
    aa_types = jnp.array(
        [AA_ONE_TO_IDX.get(r.aa_one, 0) for r in residues], dtype=jnp.int32
    )
    sasa = jnp.array(
        [r.sasa for r in residues], dtype=jnp.float32
    )
    return coords, aa_types, sasa
