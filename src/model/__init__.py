"""Inverse-folding transformer model sub-package — JAX/Flax version."""

from .transformer import (
    InverseFoldingTransformer,
    InverseFoldingLayer,
    EpitopePairEncoder,
    CDRPositionEmbedding,
    SinusoidalPositionEmbedding,
    AA_VOCAB_SIZE,
)

__all__ = [
    "InverseFoldingTransformer",
    "InverseFoldingLayer",
    "EpitopePairEncoder",
    "CDRPositionEmbedding",
    "SinusoidalPositionEmbedding",
    "AA_VOCAB_SIZE",
]
