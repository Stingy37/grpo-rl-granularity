"""Console dashboard: one fixed-width row per GRPO step, a header every 20 rows, and health
warnings when the numbers look like a stalled or collapsed run.

Columns: step | reward (mean, std) | Q (judge) | c (confidence) | pen | srch% (rollouts with >=1
search) | n_srch (mean searches) | ans% (rollouts that answered after </think>) | cut% (thinking cut
off at its token budget) | gen (mean generated tokens) | kl | pg_loss | clip (clipfrac) | ent (token
entropy) | gnorm | zeroG (fraction of groups with identical rewards) | gap (|logp_hf - logp_vllm|) |
t_roll | t_train | t_step (seconds)
"""
from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

from src.config import TrainConfig
from src.train_harness.logging import FILE_ONLY_LOGGER, MetricsLogger

log = logging.getLogger(__name__)
flog = logging.getLogger(FILE_ONLY_LOGGER)

# (record key, column title, width, format spec)
COLUMNS = [
    ("step", "step", 5, "d"),
    ("reward_mean", "reward", 7, ".3f"),
    ("reward_std", "±", 5, ".3f"),
    ("quality_mean", "Q", 6, ".3f"),
    ("confidence_mean", "c", 6, ".3f"),
    ("penalty_mean", "pen", 6, ".3f"),
    ("search_rate", "srch%", 5, ".1f"),
    ("n_searches_mean", "n_srch", 6, ".2f"),
    ("answer_rate", "ans%", 5, ".1f"),
    ("think_truncated_rate", "cut%", 5, ".1f"),
    ("gen_tokens_mean", "gen", 5, ".0f"),
    ("kl", "kl", 8, ".5f"),
    ("pg_loss", "pg_loss", 8, ".4f"),
    ("clipfrac", "clip", 5, ".3f"),
    ("entropy", "ent", 5, ".2f"),
    ("grad_norm", "gnorm", 6, ".2f"),
    ("frac_zero_var_groups", "zeroG", 5, ".2f"),
    ("lp_gap_hf_vllm", "gap", 6, ".3f"),
    ("t_rollout", "t_roll", 6, ".0f"),
    ("t_train", "t_train", 7, ".0f"),
    ("t_step", "t_step", 6, ".0f"),
]
PERCENT_KEYS = {"search_rate", "answer_rate", "think_truncated_rate"}


class Dashboard:
    def __init__(self, metrics: MetricsLogger, cfg: TrainConfig, window: int = 20, header_every: int = 20):
        self.metrics = metrics
        self.cfg = cfg
        self.window = window
        self.header_every = header_every
        self.rows = 0
        self.hist: deque[dict] = deque(maxlen=4 * window)
        for rec in metrics.history()[-4 * window:]:     # resume: carry the recent history for the health checks
            self.hist.append(rec)

    # ---- printing -------------------------------------------------------------------------
    def header(self) -> str:
        return " ".join(f"{name:>{width}}" for _, name, width, _ in COLUMNS)

    def row(self, rec: dict) -> str:
        cells = []
        for key, _name, width, spec in COLUMNS:
            v = rec.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append("-".rjust(width))
                continue
            if key in PERCENT_KEYS:
                v = 100.0 * v
            try:
                cells.append(f"{int(v) if spec == 'd' else float(v):>{width}{spec}}")
            except (ValueError, TypeError):
                cells.append(str(v).rjust(width))
        return " ".join(cells)

    def report_step(self, rec: dict) -> None:
        if self.rows % self.header_every == 0:
            print(self.header(), flush=True)
            flog.info(self.header())
        row = self.row(rec)
        print(row, flush=True)
        flog.info(row)
        self.rows += 1
        self.hist.append(rec)
        for w in self.health_warnings():
            print(f"  WARN step {rec.get('step')}: {w}", flush=True)
            flog.warning("step %s: %s", rec.get("step"), w)

    def report_eval(self, rec: dict) -> None:
        line = (f"EVAL step {rec['step']:>5d} | n={rec['n']} | EM {rec['em']:.3f} | F1 {rec['f1']:.3f} | Q {rec['quality_mean']:.3f} | "
                f"reward {rec['reward_mean']:.3f} | ans% {100 * rec['answer_rate']:.1f} | cut% {100 * rec['think_truncated_rate']:.1f} | "
                f"srch% {100 * rec['search_rate']:.1f} | "
                f"n_srch {rec['n_searches_mean']:.2f} | any-gold {rec['retrieval_any_gold']:.3f} | all-gold {rec['retrieval_all_gold']:.3f} | "
                f"{rec['eval_s']:.0f}s" + ("  ** new best **" if rec.get("is_best") else ""))
        print(line, flush=True)
        flog.info(line)

    # ---- health -----------------------------------------------------------------------------
    def health_warnings(self) -> list[str]:
        h = list(self.hist)
        if len(h) < 3:
            return []
        warns = []
        last = h[-1]
        step = last.get("step", 0)
        recent = h[-min(self.window, len(h)):]

        def series(key):
            return np.array([r.get(key) for r in recent if r.get(key) is not None and not (isinstance(r.get(key), float) and math.isnan(r.get(key)))], dtype=float)

        rs = series("reward_std")
        if len(rs) >= 5 and np.all(rs[-5:] < 0.02):
            warns.append("reward variance within batches ~0 for 5 steps: no learning signal (rewards collapsed to a constant)")
        zg = series("frac_zero_var_groups")
        if len(zg) >= 5 and np.mean(zg[-5:]) > 0.75:
            warns.append(f"{100 * np.mean(zg[-5:]):.0f}% of groups have identical rewards over the last 5 steps")
        sr = series("search_rate")
        if step > 10 and len(sr) >= 5:
            if np.all(sr[-5:] < 0.05):
                warns.append("search rate < 5% for 5 steps: policy stopped searching")
            elif np.all(sr[-5:] > 0.98):
                warns.append("search rate > 98% for 5 steps: policy always searches (check w_over / q_gate)")
        ar = series("answer_rate")
        if len(ar) >= 5 and np.all(ar[-5:] < 0.3):
            warns.append("fewer than 30% of rollouts answer after </think>: check MAX_TOKENS / THINK_FRACTION or format collapse")
        cut = series("think_truncated_rate")
        if len(cut) >= 5 and np.all(cut[-5:] > 0.9):
            warns.append("thinking hits its token budget in > 90% of rollouts for 5 steps: the policy is not compressing (or MAX_TOKENS is too small)")
        cm = series("confidence_mean")
        if len(cm) >= 3 and np.all(cm[-3:] > 0.97):
            warns.append("mean token confidence > 0.97: policy nearly deterministic (entropy collapse)")
        ent = series("entropy")
        if len(ent) >= 3 and np.all(ent[-3:] < 0.1):
            warns.append("token entropy < 0.1 for 3 steps: entropy collapse")
        kl = series("kl")
        if len(kl) >= 6:
            med = np.median(kl[:-1])
            if kl[-1] > max(1.0, 10 * med):
                warns.append(f"KL spike: {kl[-1]:.4f} vs running median {med:.4f}")
        gn = series("grad_norm")
        if len(gn) and (not np.isfinite(gn[-1]) or gn[-1] > 50):
            warns.append(f"gradient norm {gn[-1]:.2f} (pre-clip) is unusually large")
        gap = series("lp_gap_hf_vllm")
        if len(gap) and gap[-1] > 0.5:
            warns.append(f"|logp_hf - logp_vllm| = {gap[-1]:.3f}: large train/inference mismatch (adapter not loaded in vLLM?)")
        rm = series("reward_mean")
        if len(h) >= 2 * self.window:
            prev = np.array([r.get("reward_mean", np.nan) for r in h[-2 * self.window:-self.window]], dtype=float)
            if np.nanstd(np.concatenate([prev, rm])) > 0 and abs(np.nanmean(rm) - np.nanmean(prev)) < 0.005 and step > 3 * self.window:
                warns.append(f"reward plateau: mean {np.nanmean(rm):.3f} over the last {self.window} steps vs {np.nanmean(prev):.3f} before")
        return warns
