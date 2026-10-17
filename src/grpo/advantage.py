"""Group-relative advantages (GRPO, Shao et al. 2024).

Each group holds the rollouts of one prompt. A rollout's advantage is its reward minus the
group mean, optionally divided by the group standard deviation; the same scalar is applied to
every generated token of the rollout (outcome supervision, no critic).
"""
from __future__ import annotations

import numpy as np


def group_advantages(rewards: np.ndarray, groups: np.ndarray, norm_std: bool = True, eps: float = 1e-6) -> tuple[np.ndarray, dict]:
    """rewards (B,), groups (B,) integer group id -> advantages (B,) and diagnostics."""
    rewards = np.asarray(rewards, dtype=np.float64)
    groups = np.asarray(groups)
    adv = np.zeros_like(rewards)
    stds, zero_var = [], 0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        r = rewards[idx]
        mu = r.mean()
        sd = r.std() if len(r) > 1 else 0.0
        stds.append(sd)
        if sd < eps:
            zero_var += 1                     # identical rewards: no learning signal from this group
            adv[idx] = 0.0
            continue
        adv[idx] = (r - mu) / (sd + eps) if norm_std else (r - mu)
    n_groups = len(stds)
    stats = {
        "adv_mean": float(adv.mean()) if len(adv) else 0.0,
        "adv_std": float(adv.std()) if len(adv) else 0.0,
        "adv_abs_mean": float(np.abs(adv).mean()) if len(adv) else 0.0,
        "group_reward_std_mean": float(np.mean(stds)) if stds else 0.0,
        "frac_zero_var_groups": zero_var / n_groups if n_groups else 0.0,
        "n_groups": n_groups,
    }
    return adv, stats
