"""Reward = judge quality Q minus a confidence-gated penalty, as defined in the toy train_grpo.py.

  c           mean softmax probability the policy assigned to the tokens it generated (the
              rollout logprobs from vLLM; inserted <information> tokens are excluded)
  did_search  whether the trajectory issued at least one <search>
  Q           judge score in [0, 1]  (src/grpo/judge.py)

  no search  -> under-confidence penalty  w_under * (1 - ramp(c, under_lo, under_hi))
                (answering from memory with low confidence should have triggered a search)
  search     -> if Q >= q_gate: over-confidence penalty  w_over * ramp(c, over_lo, over_hi)^power
                (a good, well-grounded answer should not come with saturated confidence)
                else 0
  reward     =  Q - penalty  [- format_penalty if no <answer> was produced; off by default]
"""
from __future__ import annotations

import math

import numpy as np

from src.config import TrainConfig
from src.engine.rollout import Trajectory


def ramp(x: float, lo: float, hi: float, power: float = 1) -> float:
    """Normalised, clamped polynomial ramp between lo and hi."""
    t = min(1.0, max(0.0, (x - lo) / (hi - lo)))
    return t ** power


def compute_confidence_penalty(c: float, did_search: bool, Q: float, cfg: TrainConfig) -> float:
    if not did_search:
        return cfg.w_under * (1.0 - ramp(c, cfg.under_lo, cfg.under_hi, power=1))
    if Q >= cfg.q_gate:
        return cfg.w_over * ramp(c, cfg.over_lo, cfg.over_hi, power=cfg.pen_power)
    return 0.0


def token_confidence(traj: Trajectory) -> float:
    """Average probability of the generated tokens (the toy's compute_token_confidence, taken from
    the rollout logprobs instead of a second forward pass)."""
    lp = np.asarray(traj.gen_logprobs, dtype=np.float64)
    m = np.asarray(traj.response_mask, dtype=bool)
    if not m.any():
        return 0.0
    return float(np.exp(lp[m]).mean())


def compute_rewards(trajs: list[Trajectory], qualities: list[float], cfg: TrainConfig) -> np.ndarray:
    """Fill confidence / quality / penalty / reward on each trajectory; return the reward vector."""
    rewards = np.empty(len(trajs), dtype=np.float64)
    for i, (t, Q) in enumerate(zip(trajs, qualities)):
        c = token_confidence(t)
        Q = float(min(1.0, max(0.0, Q))) if not math.isnan(Q) else 0.5
        pen = compute_confidence_penalty(c, t.did_search, Q, cfg)
        r = Q - pen
        if cfg.format_penalty and t.answer is None:
            r -= cfg.format_penalty
        t.confidence, t.quality, t.penalty, t.reward = c, Q, pen, r
        rewards[i] = r
    return rewards
