"""GRPO objective: clipped policy-gradient surrogate + KL penalty to the reference policy.

    ratio_t = exp(logp_new_t - logp_old_t)
    L_pg    = -min(ratio_t * A, clip(ratio_t, 1-eps, 1+eps) * A)
    L_kl    = kl_coef * KL_t(ref || new)         low_var_kl (k3, Search-R1/verl default) | kl | mse
    L       = sum_t mask_t * (L_pg + L_kl) / normaliser

The normaliser is supplied by the caller so gradient accumulation over micro-batches is exact:
token-mean divides by the number of trained tokens in the minibatch, seq-mean-token-mean by
the number of sequences (after a per-sequence token mean).
"""
from __future__ import annotations

import torch


def kl_penalty(new_lp: torch.Tensor, ref_lp: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "low_var_kl":
        d = (ref_lp - new_lp).clamp(-20.0, 20.0)
        return (torch.exp(d) - d - 1.0).clamp(-10.0, 10.0)
    if kind == "kl":
        return new_lp - ref_lp
    if kind == "mse":
        return 0.5 * (new_lp - ref_lp) ** 2
    raise ValueError(kind)


def grpo_loss(new_lp: torch.Tensor, old_lp: torch.Tensor, ref_lp: torch.Tensor | None, adv: torch.Tensor, mask: torch.Tensor,
              clip_eps: float, kl_coef: float, kl_type: str, agg: str, normaliser: float) -> tuple[torch.Tensor, dict]:
    """All tensors (B, T) except adv (B,). Returns (loss to backward, detached token-weighted stats)."""
    mask = mask.to(new_lp.dtype)
    adv_t = adv.to(new_lp.dtype).unsqueeze(1)
    log_ratio = new_lp - old_lp
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pg = -torch.min(ratio * adv_t, clipped * adv_t)
    per_tok = pg
    kl = None
    if ref_lp is not None and kl_coef > 0:
        kl = kl_penalty(new_lp, ref_lp, kl_type)
        per_tok = per_tok + kl_coef * kl

    if agg == "seq-mean-token-mean":
        seq = (per_tok * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        loss = seq.sum() / normaliser
    else:
        loss = (per_tok * mask).sum() / normaliser

    with torch.no_grad():
        n = mask.sum().clamp(min=1.0)
        clipfrac = (((ratio - 1.0).abs() > clip_eps).to(mask.dtype) * mask).sum() / n
        stats = {
            "pg_loss": float((pg * mask).sum() / n),
            "kl": float((kl * mask).sum() / n) if kl is not None else 0.0,
            "clipfrac": float(clipfrac),
            "ratio_mean": float((ratio * mask).sum() / n),
            "approx_kl_old": float((0.5 * log_ratio ** 2 * mask).sum() / n),
            "n_tokens": float(n),
        }
    return loss, stats
