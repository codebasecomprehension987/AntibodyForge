"""
scripts/design.py  (JAX)
=========================
Inference CLI — JAX version. Same interface as the PyTorch version.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.pipeline import AntibodyForgePipeline, ForgeConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AntibodyForge (JAX): de novo CDR-H3 inverse-folding design"
    )
    p.add_argument("--pdb",            required=True,  type=str)
    p.add_argument("--heavy-chain",    default="H",    type=str)
    p.add_argument("--antigen-chains", nargs="+",      default=None)
    p.add_argument("--config",         default="configs/default.yaml", type=str)
    p.add_argument("--checkpoint",     default=None,   type=str)
    p.add_argument("--top-k",          default=10,     type=int)
    p.add_argument("--output",         default="designs/output.fasta", type=str)
    p.add_argument("--beam-width",     default=None,   type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    overrides = {}
    if args.beam_width is not None:
        overrides["beam_width"] = args.beam_width

    pipeline = AntibodyForgePipeline.from_config(
        args.config,
        checkpoint_path=args.checkpoint,
        **overrides,
    )

    results = pipeline.design(
        pdb_path          = args.pdb,
        heavy_chain_id    = args.heavy_chain,
        antigen_chain_ids = args.antigen_chains,
        top_k             = args.top_k,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    AntibodyForgePipeline.results_to_fasta(results, out_path)

    print(f"\n{'='*60}")
    print(f"  AntibodyForge (JAX) — Top {args.top_k} CDR-H3 Designs")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        print(f"  {i:>3}. {r.sequence:<30}  logP={r.logprob:+.3f}  ΔG={r.delta_g:+.2f} kcal/mol")
    print(f"{'='*60}\n  FASTA → {out_path}\n")


if __name__ == "__main__":
    main()
