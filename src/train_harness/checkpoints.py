"""Periodic held-out evaluation of the current policy (vLLM + adapter, greedy decoding) and
checkpoint management.

output/rl/checkpoints/
  step_NNNNN/adapter/      LoRA adapter as saved by PEFT (loadable by vLLM and by LocalPolicy)
  step_NNNNN/metrics.json  held-out metrics of that adapter
  step_NNNNN/samples.jsonl the evaluated trajectories (question, answer, gold, searches, text)
  best/                    copy of the step with the best BEST_METRIC so far (+ metrics.json)
Old step_* directories are pruned down to KEEP_LAST, best/ is never pruned.

Metrics: exact match and token F1 (HotpotQA normalisation), answer rate, search rate, mean
number of searches, retrieval any-/all-gold coverage (gold article titles among everything
retrieved in the trajectory), judge Q, the training reward, mean generated / thinking tokens,
how often the thinking hit its token budget.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.config import TrainConfig
from src.engine.rollout import new_trajectories, run_rollouts
from src.engine.vllm import VLLMEngine
from src.grpo.judge import Judge
from src.grpo.reward import compute_rewards
from src.retriever.dense_retriever import DenseRetriever, gold_hits
from src.train_harness.logging import MetricsLogger

log = logging.getLogger(__name__)


# ---- HotpotQA official normalisation ----------------------------------------------------------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str | None, gold: str) -> float:
    if pred is None:
        return 0.0
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str | None, gold: str) -> float:
    if pred is None:
        return 0.0
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer"):
        return float(p == g)
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(pt), ns / len(gt)
    return 2 * prec * rec / (prec + rec)


class Evaluator:
    def __init__(self, cfg: TrainConfig, engine: VLLMEngine, retriever: DenseRetriever, judge: Judge, heldout: list[dict],
                 metrics: MetricsLogger):
        self.cfg = cfg
        self.engine = engine
        self.retriever = retriever
        self.judge = judge
        self.heldout = heldout
        self.metrics = metrics
        self.best_value = -float("inf")
        self.best_step = -1
        best_meta = cfg.ckpt_dir / "best" / "metrics.json"
        if best_meta.exists():
            try:
                m = json.loads(best_meta.read_text())
                self.best_value, self.best_step = float(m.get(self._metric_key(), -float("inf"))), int(m.get("step", -1))
                log.info("best checkpoint so far: step %d %s=%.4f", self.best_step, cfg.best_metric, self.best_value)
            except Exception:
                pass

    def _metric_key(self) -> str:
        return {"em": "em", "f1": "f1", "reward": "reward_mean"}[self.cfg.best_metric]

    def evaluate(self, step: int, adapter_path: Path | None) -> dict:
        """Rollouts of the policy currently loaded in vLLM over the held-out set, with the EVAL_* sampling
        settings and fixed seeds (Qwen3 advises against greedy decoding in thinking mode)."""
        cfg = self.cfg
        t0 = time.perf_counter()
        trajs = new_trajectories(self.engine, self.heldout, 1)
        rstats = run_rollouts(self.engine, self.retriever, trajs, cfg, eval_mode=True, use_lora=adapter_path is not None)
        Q = self.judge.score(trajs)
        rewards = compute_rewards(trajs, Q, cfg)
        em = np.array([exact_match(t.answer, t.gold_answer) for t in trajs])
        f1 = np.array([f1_score(t.answer, t.gold_answer) for t in trajs])
        any_gold, all_gold = [], []
        for t in trajs:
            seen = [p for ps in t.retrieved for p in ps]
            hit, n = gold_hits(t.gold_titles, seen)
            any_gold.append(float(hit > 0))
            all_gold.append(float(n > 0 and hit == n))
        finish = Counter(t.finish for t in trajs)
        rec = {
            "step": step, "n": len(trajs), "em": float(em.mean()), "f1": float(f1.mean()),
            "answer_rate": float(np.mean([t.answer is not None for t in trajs])),
            "search_rate": float(np.mean([t.did_search for t in trajs])),
            "n_searches_mean": float(np.mean([t.n_searches for t in trajs])),
            "retrieval_any_gold": float(np.mean(any_gold)), "retrieval_all_gold": float(np.mean(all_gold)),
            "quality_mean": float(Q.mean()), "reward_mean": float(rewards.mean()),
            "confidence_mean": float(np.mean([t.confidence for t in trajs])),
            "penalty_mean": float(np.mean([t.penalty for t in trajs])),
            "gen_tokens_mean": float(np.mean([t.n_generated for t in trajs])),
            "think_tokens_mean": float(np.mean([t.think_tokens for t in trajs])),
            "think_truncated_rate": float(np.mean([t.think_truncated for t in trajs])),
            "answer_in_think_rate": float(np.mean([t.finish == "answer_in_think" for t in trajs])),
            "finish": dict(finish), "eval_s": time.perf_counter() - t0, "rollout_generate_s": rstats.generate_s,
            "rollout_retrieval_s": rstats.retrieval_s, "adapter": str(adapter_path) if adapter_path else None,
        }
        value = rec[self._metric_key()]
        rec["is_best"] = value > self.best_value
        self.metrics.log_eval(rec)
        self._save_checkpoint(step, adapter_path, rec, trajs)
        if rec["is_best"]:
            self.best_value, self.best_step = value, step
        return rec

    # ---- checkpoint files -----------------------------------------------------------------
    def _save_checkpoint(self, step: int, adapter_path: Path | None, rec: dict, trajs: list) -> None:
        d = self.cfg.ckpt_dir / f"step_{step:05d}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        if adapter_path and Path(adapter_path).exists():
            shutil.copytree(adapter_path, d / "adapter", ignore=shutil.ignore_patterns("optimizer.pt"))
        (d / "metrics.json").write_text(json.dumps(rec, indent=2, default=str))
        with open(d / "samples.jsonl", "w") as f:
            for t in trajs:
                f.write(json.dumps({**t.summary(), "question": t.question, "searches": t.searches,
                                    "retrieved_titles": [[p.title for p in ps] for ps in t.retrieved], "text": t.text},
                                   ensure_ascii=False) + "\n")
        if rec["is_best"]:
            best = self.cfg.ckpt_dir / "best"
            if best.exists():
                shutil.rmtree(best)
            shutil.copytree(d, best)
            log.info("new best %s=%.4f at step %d -> %s", self.cfg.best_metric, rec[self._metric_key()], step, best)
        self._prune()

    def _prune(self) -> None:
        steps = sorted(p for p in self.cfg.ckpt_dir.glob("step_*") if p.is_dir())
        for p in steps[:-self.cfg.keep_last] if self.cfg.keep_last > 0 else []:
            shutil.rmtree(p, ignore_errors=True)
