"""
scripts/train.py  (JAX)
========================
Training script — JAX/Flax/Optax rewrite.

PyTorch → JAX training loop translation
-----------------------------------------

  PyTorch                          JAX / Flax / Optax
  ──────────────────────────────────────────────────────────────────
  model.train()                  → deterministic=False in model.apply
  model.eval()                   → deterministic=True  in model.apply
  optimizer = AdamW(model.params)→ optax.adamw(lr)
  optimizer.zero_grad()          → (implicit — JAX grad is functional)
  loss.backward()                → jax.value_and_grad(loss_fn)(params)
  optimizer.step()               → optax.apply_updates(params, grads)
  scheduler.step()               → optax.cosine_decay_schedule
  torch.save / torch.load        → orbax.checkpoint
  torch.no_grad()                → (not needed — JAX is grad-free by default)

Functional training step
------------------------
JAX training follows a purely functional pattern:

    def train_step(params, opt_state, batch, key):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, key
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

This is then wrapped in ``jax.jit`` for XLA compilation.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Optional
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.pipeline import ForgeConfig, AntibodyForgePipeline
from src.model.transformer import InverseFoldingTransformer, AA_VOCAB_SIZE
from src.utils.structure import (
    parse_pdb_chains, detect_cdr_h3, compute_epitope_adjacency,
    build_graph_tensors, residues_to_jax, AA_ONE_TO_IDX,
)
from src.kernels.sparse_cdr_attention import MAX_CDR_LEN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("antibodyforge_jax.train")

BOS_ID = 1
EOS_ID = 2


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AntibodyForge (JAX)")
    p.add_argument("--config",     type=str, default="configs/default.yaml")
    p.add_argument("--data",       type=str, default="data/train.jsonl")
    p.add_argument("--val-data",   type=str, default="data/val.jsonl")
    p.add_argument("--output-dir", type=str, default="checkpoints_jax")
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--resume",     type=str, default=None)
    p.add_argument("--log-every",  type=int, default=50)
    p.add_argument("--save-every", type=int, default=500)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def encode_seq(seq: str) -> List[int]:
    ids = [BOS_ID]
    for aa in seq.upper():
        idx = AA_ONE_TO_IDX.get(aa)
        if idx is not None:
            ids.append(idx + 3)
    ids.append(EOS_ID)
    return ids


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_loss(
    params,
    model: InverseFoldingTransformer,
    token_ids:  jnp.ndarray,   # [MAX_CDR_LEN]
    ag_coords:  jnp.ndarray,
    ag_types:   jnp.ndarray,
    ag_sasa:    jnp.ndarray,
    row_ptr:    jnp.ndarray,
    col_idx:    jnp.ndarray,
    valid_mask: jnp.ndarray,
    lengths:    jnp.ndarray,
    dropout_key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """
    Teacher-forced cross-entropy loss.

    input  = token_ids[:-1]  (shifted right by 1)
    target = token_ids[1:]
    """
    T = token_ids.shape[0]

    log_probs = model.apply(
        {"params": params},
        token_ids, ag_coords, ag_types, ag_sasa,
        row_ptr, col_idx, valid_mask, lengths,
        deterministic=False,
        rngs={"dropout": dropout_key},
    )  # [MAX_CDR_LEN, VOCAB]

    # Compute cross-entropy only over valid positions
    # target[t] = token_ids[t+1] for t in 0..T-2
    targets   = jnp.roll(token_ids, -1)            # [MAX_CDR_LEN]
    # valid positions: where valid_mask is True and not the last position
    pos_mask  = valid_mask & jnp.arange(MAX_CDR_LEN) < (lengths - 1)

    nll = -log_probs[jnp.arange(MAX_CDR_LEN), targets]  # [MAX_CDR_LEN]
    loss = jnp.sum(nll * pos_mask) / jnp.maximum(jnp.sum(pos_mask), 1)
    return loss


# ---------------------------------------------------------------------------
# JIT-compiled train step
# ---------------------------------------------------------------------------

def make_train_step(model: InverseFoldingTransformer, optimizer: optax.GradientTransformation):
    """Return a JIT-compiled training step function."""

    @jax.jit
    def train_step(params, opt_state, batch: dict, key: jax.random.PRNGKey):
        key, dropout_key = jax.random.split(key)

        loss_val, grads = jax.value_and_grad(compute_loss)(
            params, model,
            batch["token_ids"],
            batch["ag_coords"],
            batch["ag_types"],
            batch["ag_sasa"],
            batch["row_ptr"],
            batch["col_idx"],
            batch["valid_mask"],
            batch["lengths"],
            dropout_key,
        )

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val, key

    return train_step


# ---------------------------------------------------------------------------
# Build batch from record
# ---------------------------------------------------------------------------

def record_to_batch(
    record: dict,
    cfg: ForgeConfig,
) -> Optional[dict]:
    """Convert a JSONL record to a model-ready batch dict."""
    try:
        chains = parse_pdb_chains(record["pdb"])
    except Exception:
        return None

    heavy      = chains.get(record.get("heavy_chain", "H"), [])
    ag_ids     = record.get("antigen_chains", None)
    if not heavy:
        return None

    if ag_ids is None:
        ag_ids = [c for c in chains if c != record.get("heavy_chain", "H")]
    ag_res = []
    for cid in ag_ids:
        ag_res.extend(chains.get(cid, []))
    if not ag_res:
        return None

    cdr_res = detect_cdr_h3(heavy, scheme=cfg.numbering_scheme)
    if not cdr_res:
        return None

    cdr_len, n_ag, edges = compute_epitope_adjacency(
        cdr_res, ag_res, cutoff=cfg.epitope_cutoff_A
    )
    row_ptr, col_idx, valid_mask = build_graph_tensors(cdr_len, n_ag, edges)
    ag_coords, ag_types, ag_sasa = residues_to_jax(ag_res)

    ids  = encode_seq(record["cdr_h3_seq"])
    # Pad to MAX_CDR_LEN
    padded = ids[:MAX_CDR_LEN] + [0] * max(0, MAX_CDR_LEN - len(ids))
    token_ids = jnp.array(padded, dtype=jnp.int32)
    lengths   = jnp.array(min(len(ids), MAX_CDR_LEN), dtype=jnp.int32)

    return {
        "token_ids":  token_ids,
        "ag_coords":  ag_coords,
        "ag_types":   ag_types,
        "ag_sasa":    ag_sasa,
        "row_ptr":    row_ptr,
        "col_idx":    col_idx,
        "valid_mask": valid_mask,
        "lengths":    lengths,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg    = ForgeConfig.from_yaml(args.config)
    cfg.seed = args.seed

    logger.info("Loading training data: %s", args.data)
    train_records = load_jsonl(args.data)
    val_records   = load_jsonl(args.val_data) if Path(args.val_data).exists() else []
    logger.info("Train: %d | Val: %d", len(train_records), len(val_records))

    model = InverseFoldingTransformer(
        d_model      = cfg.d_model,
        n_heads      = cfg.n_heads,
        n_layers     = cfg.n_layers,
        ffn_dim      = cfg.ffn_dim,
        dropout_rate = cfg.dropout_rate,
    )

    # Build Optax optimiser with cosine decay schedule
    # This replaces PyTorch's AdamW + CosineAnnealingLR
    total_steps = args.epochs * max(len(train_records), 1)
    schedule    = optax.cosine_decay_schedule(
        init_value  = args.lr,
        decay_steps = total_steps,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),       # gradient clipping
        optax.adamw(learning_rate=schedule, weight_decay=1e-4),
    )

    # Initialise parameters via pipeline helper
    pipeline   = AntibodyForgePipeline(cfg)
    params     = pipeline.params
    opt_state  = optimizer.init(params)

    key = jax.random.PRNGKey(args.seed)

    if args.resume:
        try:
            import orbax.checkpoint as ocp
            checkpointer = ocp.PyTreeCheckpointer()
            params = checkpointer.restore(args.resume, item=params)
            logger.info("Resumed from %s", args.resume)
        except Exception as e:
            logger.warning("Could not load checkpoint: %s", e)

    train_step = make_train_step(model, optimizer)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0

    for epoch in range(args.epochs):
        random.shuffle(train_records)
        epoch_loss = 0.0
        n_valid    = 0

        for rec in train_records:
            batch = record_to_batch(rec, cfg)
            if batch is None:
                continue

            params, opt_state, loss_val, key = train_step(
                params, opt_state, batch, key
            )

            epoch_loss  += float(loss_val)
            n_valid     += 1
            global_step += 1

            if global_step % args.log_every == 0:
                logger.info(
                    "Epoch %d | Step %d | Loss %.4f",
                    epoch, global_step, epoch_loss / max(n_valid, 1),
                )

            if global_step % args.save_every == 0:
                try:
                    import orbax.checkpoint as ocp
                    ckpt_path = output_dir / f"step_{global_step:07d}"
                    ocp.PyTreeCheckpointer().save(str(ckpt_path), params)
                    logger.info("Checkpoint: %s", ckpt_path)
                except ImportError:
                    pass

        logger.info(
            "=== Epoch %d | Loss %.4f ===",
            epoch, epoch_loss / max(n_valid, 1),
        )

    # Final checkpoint
    try:
        import orbax.checkpoint as ocp
        ocp.PyTreeCheckpointer().save(str(output_dir / "final"), params)
        logger.info("Training complete. Final checkpoint saved.")
    except ImportError:
        logger.warning("orbax not installed; final params not saved")


if __name__ == "__main__":
    main()
