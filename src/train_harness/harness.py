"""The training loop. One GRPO step:

  1. sample NUM_GROUPS HotpotQA questions, ROLLOUTS_PER_GROUP starts each        (sampling.py)
  2. rollouts on vLLM (fp8 base + current LoRA): native thinking with searches  (engine/rollout.py)
     inside the block, token budgets instead of turn caps
  3. judge Q and reference log-probs on the same vLLM, adapter off              (grpo/judge.py, engine/vllm.py)
  4. reward = Q - confidence penalty; group-normalised advantages               (grpo/reward.py, grpo/advantage.py)
  5. vLLM -> sleep (weights to CPU RAM); HF QLoRA model -> GPU                  (engine/vllm.py, engine/local_llm.py)
  6. clipped PG + KL update, PPO_EPOCHS passes; save the adapter                (grpo/backprop.py)
  7. HF model -> CPU; vLLM -> wake; point rollouts at the new adapter
  8. log + print; every EVAL_EVERY steps evaluate greedily on the held-out set  (dashboard.py, checkpoints.py)

Only one of the two 8B models is resident on the 33 GB MIG slice at any time.
"""
from __future__ import annotations

import json
import logging
import math
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.config import TrainConfig
from src.engine.local_llm import LocalPolicy
from src.engine.rollout import new_trajectories, run_rollouts
from src.engine.vllm import VLLMEngine, free_gpu_gb
from src.grpo.advantage import group_advantages
from src.grpo.backprop import GRPOUpdater
from src.grpo.judge import Judge
from src.grpo.reward import compute_rewards
from src.retriever.dense_retriever import DenseRetriever
from src.train_harness.checkpoints import Evaluator
from src.train_harness.dashboard import Dashboard
from src.train_harness.logging import MetricsLogger, setup_logging
from src.train_harness.sampling import GroupSampler, build_pools

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Harness:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        cfg.validate()
        cfg.ensure_dirs()
        self.logfile = setup_logging(cfg.log_dir, cfg.log_level)
        cfg.dump(cfg.output_dir / "config.json")
        set_seed(cfg.seed)
        self.metrics = MetricsLogger(cfg.log_dir)
        self.dash = Dashboard(self.metrics, cfg)
        self.state_path = cfg.output_dir / "state.json"
        log.info("config: %s", json.dumps(cfg.to_dict()))

    # ---- setup ----------------------------------------------------------------------------
    def setup(self) -> None:
        cfg = self.cfg
        state = self._load_state()
        self.start_step = int(state.get("step", 0))
        adapter = cfg.resume_from or state.get("adapter") or None
        if adapter and not Path(adapter).exists():
            raise FileNotFoundError(f"resume adapter {adapter}")

        train_pool, heldout = build_pools(cfg)
        self.sampler = GroupSampler(train_pool, cfg.seed)
        if state.get("sampler"):
            self.sampler.load_state(state["sampler"])

        log.info("free GPU before setup: %.1f GB", free_gpu_gb())
        self.retriever = DenseRetriever(cfg)                                   # faiss on CPU, e5 on GPU (~0.3 GB)
        self.engine = VLLMEngine(cfg)                                          # fp8 Qwen3-8B
        self.judge = Judge(self.engine, cfg)

        # the HF side is built while vLLM sleeps so the quantised load has the GPU to itself
        self.engine.sleep()
        n_mb = math.ceil(cfg.batch_size / (cfg.minibatch_size or cfg.batch_size))
        self.policy = LocalPolicy(cfg, adapter_path=adapter, total_optimizer_steps=cfg.total_steps * cfg.ppo_epochs * n_mb)
        self.policy.swap_check()
        if adapter and (Path(adapter) / "optimizer.pt").exists():
            self.policy.load_optimizer(Path(adapter) / "optimizer.pt")
        if not adapter:
            adapter = self.policy.save_adapter(cfg.adapter_dir / f"step_{self.start_step:05d}")   # zero-init B: equals the base
        self.adapter_path = Path(adapter)
        self.policy.offload()
        self.engine.wake()
        self.engine.set_adapter(str(self.adapter_path), tag=str(self.start_step))
        self.updater = GRPOUpdater(self.policy, cfg)
        self.evaluator = Evaluator(cfg, self.engine, self.retriever, self.judge, heldout, self.metrics)
        log.info("setup done: start step %d, adapter %s, free GPU %.1f GB", self.start_step, self.adapter_path, free_gpu_gb())

    # ---- loop -----------------------------------------------------------------------------
    def run(self) -> None:
        cfg = self.cfg
        self.setup()
        if self.start_step == 0 and cfg.eval_every > 0:
            self.dash.report_eval(self.evaluator.evaluate(0, self.adapter_path))

        for step in range(self.start_step, cfg.total_steps):
            t_step = time.perf_counter()
            n = step + 1
            # 1-2. rollouts
            items = self.sampler.sample(cfg.num_groups)
            trajs = new_trajectories(self.engine, items, cfg.rollouts_per_group)
            seed = (cfg.seed * 7919 + step) if cfg.rollout_seed_per_step else None
            rstats = run_rollouts(self.engine, self.retriever, trajs, cfg, seed=seed, use_lora=True)
            t_roll = time.perf_counter()

            # 3. judge + reference log-probs (adapter off)
            Q = self.judge.score(trajs)
            t_judge = time.perf_counter()
            ref = None
            if cfg.kl_coef > 0 and cfg.ref_logprobs_source == "vllm":
                ref = self.engine.prompt_logprobs([t.full_ids for t in trajs], use_lora=False)
            t_ref = time.perf_counter()

            # 4. reward, advantage
            rewards = compute_rewards(trajs, Q, cfg)
            groups = np.array([t.group for t in trajs])
            adv, astats = group_advantages(rewards, groups, norm_std=cfg.adv_norm_std)
            for t, a in zip(trajs, adv):
                t.advantage = float(a)

            # 5-7. swap, update, swap back
            self.engine.sleep()
            self.policy.restore()
            t_swap1 = time.perf_counter()
            tstats = self.updater.update(trajs, ref)
            new_adapter = self.policy.save_adapter(cfg.adapter_dir / f"step_{n:05d}")
            if n % cfg.save_every == 0 or n == cfg.total_steps:
                self.policy.save_optimizer(new_adapter / "optimizer.pt")
            t_train = time.perf_counter()
            self.policy.offload()
            self.engine.wake()
            self.engine.set_adapter(str(new_adapter), tag=str(n))
            self._prune_adapters(keep=2)
            self.adapter_path = new_adapter
            t_end = time.perf_counter()

            # 8. bookkeeping
            rec = self._record(n, trajs, rewards, astats, tstats, rstats)
            rec.update({"t_rollout": t_roll - t_step, "t_judge": t_judge - t_roll, "t_ref": t_ref - t_judge,
                        "t_swap_to_hf": t_swap1 - t_ref, "t_train": t_train - t_swap1, "t_swap_to_vllm": t_end - t_train,
                        "t_step": t_end - t_step, "free_gpu_gb": free_gpu_gb()})
            self.metrics.log_step(rec)
            self.dash.report_step(rec)
            if cfg.sample_dump_every and n % cfg.sample_dump_every == 0:
                self.metrics.dump_samples(n, trajs)
            self._save_state(n)
            if cfg.eval_every and n % cfg.eval_every == 0:
                self.dash.report_eval(self.evaluator.evaluate(n, self.adapter_path))

        log.info("done: %d steps, best %s=%.4f at step %d, logs in %s", cfg.total_steps, cfg.best_metric,
                 self.evaluator.best_value, self.evaluator.best_step, cfg.log_dir)

    # ---- helpers --------------------------------------------------------------------------
    def _record(self, step: int, trajs, rewards, astats: dict, tstats: dict, rstats) -> dict:
        et = self.engine.reset_timings()
        finish = Counter(t.finish for t in trajs)
        rec = {
            "step": step, "n_rollouts": len(trajs),
            "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std()),
            "reward_min": float(rewards.min()), "reward_max": float(rewards.max()),
            "quality_mean": float(np.mean([t.quality for t in trajs])),
            "confidence_mean": float(np.mean([t.confidence for t in trajs])),
            "confidence_std": float(np.std([t.confidence for t in trajs])),
            "penalty_mean": float(np.mean([t.penalty for t in trajs])),
            "search_rate": float(np.mean([t.did_search for t in trajs])),
            "n_searches_mean": float(np.mean([t.n_searches for t in trajs])),
            "answer_rate": float(np.mean([t.finish == "answer" for t in trajs])),
            "answer_in_think_rate": float(np.mean([t.finish == "answer_in_think" for t in trajs])),
            "think_truncated_rate": float(np.mean([t.think_truncated for t in trajs])),
            "think_tokens_mean": float(np.mean([t.think_tokens for t in trajs])),
            "answer_tokens_mean": float(np.mean([t.answer_tokens for t in trajs])),
            "think_closed": dict(Counter(t.think_closed or "never" for t in trajs)),
            "n_turns_mean": float(np.mean([t.n_turns for t in trajs])),
            "gen_tokens_mean": float(np.mean([t.n_generated for t in trajs])),
            "resp_tokens_mean": float(np.mean([len(t.response_ids) for t in trajs])),
            "resp_tokens_max": int(max(len(t.response_ids) for t in trajs)),
            "finish": dict(finish),
            "judge_parse_failures": self.judge.n_parse_failures,
            "rollout_generate_s": rstats.generate_s, "rollout_retrieval_s": rstats.retrieval_s,
            "n_retrievals": rstats.n_retrievals, "retrieval_ms_last": self.retriever.last_stats.total_ms,
            "vllm_sleep_s": et.sleep_s, "vllm_wake_s": et.wake_s, "vllm_judge_s": et.chat_s, "vllm_ref_s": et.prompt_logprobs_s,
            "sampler": self.sampler.state(),
        }
        rec.update(astats)
        rec.update(tstats)
        return rec

    def _prune_adapters(self, keep: int) -> None:
        dirs = sorted(p for p in self.cfg.adapter_dir.glob("step_*") if p.is_dir())
        for p in dirs[:-keep]:
            if not (p / "optimizer.pt").exists():        # adapters with optimizer state are resume points: keep them
                shutil.rmtree(p, ignore_errors=True)

    def _save_state(self, step: int) -> None:
        self.state_path.write_text(json.dumps({"step": step, "adapter": str(self.adapter_path), "sampler": self.sampler.state(),
                                               "best_step": self.evaluator.best_step, "best_value": self.evaluator.best_value}))

    def _load_state(self) -> dict:
        if self.cfg.resume_from or self.state_path.exists():
            try:
                st = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
            except json.JSONDecodeError:
                st = {}
            if self.cfg.resume_from:
                st["adapter"] = self.cfg.resume_from
                if "step" not in st:
                    try:
                        st["step"] = int(Path(self.cfg.resume_from).name.split("_")[-1])
                    except ValueError:
                        st["step"] = 0
            if st:
                log.info("resuming from step %s (%s)", st.get("step"), st.get("adapter"))
            return st
        return {}


def run(cfg: TrainConfig) -> None:
    Harness(cfg).run()
