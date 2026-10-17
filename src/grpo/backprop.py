"""One GRPO update on the HF QLoRA policy from a batch of scored trajectories.

    for epoch in range(PPO_EPOCHS):                      # multi-step reuse of the same rollouts
        for minibatch in shuffle(batch):                 # MINIBATCH_SIZE sequences per optimiser step
            for micro in minibatch:                      # HF_MICRO_BATCH_SIZE sequences per fwd/bwd
                new_lp = policy.logprobs(micro)          # with grad, adapter on
                loss   = grpo_loss(new_lp, old_lp, ref_lp, adv, mask) / normaliser
                loss.backward()
            clip_grad_norm, optimizer.step(), scheduler.step()

old_lp  log-probs of the sampled tokens under the policy that produced them. OLD_LOGPROBS_SOURCE=hf
        recomputes them with this model before the first step (so the ratio is exactly 1 on the
        first pass and the PPO clipping is meaningful afterwards); =vllm reuses the fp8 rollout
        logprobs. The mean |hf - vllm| gap is logged either way as the train/inference mismatch.
ref_lp  reference log-probs: from vLLM (base model, no adapter) by default, or from this model
        with the adapter disabled (REF_LOGPROBS_SOURCE=hf).
Only generated tokens (response_mask == 1) contribute; prompt and <information> tokens do not.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import torch

from src.config import TrainConfig
from src.engine.local_llm import LocalPolicy
from src.engine.rollout import Trajectory
from src.grpo.loss import grpo_loss

log = logging.getLogger(__name__)


class GRPOUpdater:
    def __init__(self, policy: LocalPolicy, cfg: TrainConfig):
        self.policy = policy
        self.cfg = cfg
        self.optimizer_steps = 0

    # ---- tensors --------------------------------------------------------------------------
    def _loss_mask(self, t: Trajectory) -> np.ndarray:
        """Mask over the shifted frame (positions 1..T-1 of full_ids): 1 where token i is a generated
        response token."""
        P = len(t.prompt_ids)
        m = np.zeros(P + len(t.response_ids) - 1, dtype=np.float32)
        m[P - 1:] = np.asarray(t.response_mask, dtype=np.float32)
        return m

    def _collate(self, trajs: list[Trajectory], extra: dict[str, list[np.ndarray]]) -> dict[str, torch.Tensor]:
        pad = self.policy.tokenizer.pad_token_id
        if pad is None:
            pad = self.policy.tokenizer.eos_token_id
        T = max(len(t.full_ids) for t in trajs)
        B = len(trajs)
        ids = torch.full((B, T), pad, dtype=torch.long)
        att = torch.zeros((B, T), dtype=torch.long)
        mask = torch.zeros((B, T - 1), dtype=torch.float32)
        extras = {k: torch.zeros((B, T - 1), dtype=torch.float32) for k in extra}
        for b, t in enumerate(trajs):
            f = t.full_ids
            ids[b, :len(f)] = torch.tensor(f, dtype=torch.long)
            att[b, :len(f)] = 1
            m = self._loss_mask(t)
            mask[b, :len(m)] = torch.from_numpy(m)
            for k, arrs in extra.items():
                a = np.nan_to_num(np.asarray(arrs[b], dtype=np.float32), nan=0.0)
                extras[k][b, :len(a)] = torch.from_numpy(a)
        dev = self.policy.device
        out = {"input_ids": ids.to(dev), "attention_mask": att.to(dev), "mask": mask.to(dev),
               "adv": torch.tensor([t.advantage for t in trajs], dtype=torch.float32, device=dev)}
        out.update({k: v.to(dev) for k, v in extras.items()})
        return out

    @torch.no_grad()
    def _nograd_logprobs(self, trajs: list[Trajectory], use_adapter: bool) -> list[np.ndarray]:
        """Shifted-frame log-probs (len T-1 per trajectory) from the HF model, no grad."""
        res: list[np.ndarray] = [None] * len(trajs)
        order = sorted(range(len(trajs)), key=lambda i: len(trajs[i].full_ids))
        mb = self.cfg.hf_micro_batch_size
        for s in range(0, len(order), mb):
            idx = order[s:s + mb]
            batch = self._collate([trajs[i] for i in idx], {})
            lp, _ = self.policy.logprobs(batch["input_ids"], batch["attention_mask"], with_grad=False, use_adapter=use_adapter)
            lp = lp.float().cpu().numpy()
            for b, i in enumerate(idx):
                res[i] = lp[b, :len(trajs[i].full_ids) - 1].copy()
            del batch, lp
        return res

    # ---- the update -----------------------------------------------------------------------
    def update(self, trajs: list[Trajectory], ref_lp_full: list[np.ndarray] | None) -> dict:
        """ref_lp_full: per trajectory, log-prob of every token of full_ids (index 0 NaN) as returned by
        VLLMEngine.prompt_logprobs; None -> computed here with the adapter disabled."""
        cfg = self.cfg
        t0 = time.perf_counter()
        B = len(trajs)
        stats: dict = {"skipped": False}
        if all(abs(t.advantage) < 1e-8 for t in trajs):
            log.warning("all advantages are zero: skipping the gradient step")
            stats["skipped"] = True
            return stats

        self.policy.train()
        # reference log-probs in the shifted frame
        if ref_lp_full is None or cfg.ref_logprobs_source == "hf":
            ref = self._nograd_logprobs(trajs, use_adapter=False)
        else:
            ref = [np.asarray(lp[1:], dtype=np.float32) for lp in ref_lp_full]

        # rollout-time (vLLM) log-probs in the shifted frame, for the ratio or the mismatch diagnostic
        vllm_old = []
        for t in trajs:
            a = np.zeros(len(t.full_ids) - 1, dtype=np.float32)
            a[len(t.prompt_ids) - 1:] = np.asarray(t.gen_logprobs, dtype=np.float32)
            vllm_old.append(a)
        if cfg.old_logprobs_source == "hf":
            old = self._nograd_logprobs(trajs, use_adapter=True)
            gaps, ents = [], []
            for t, o, v in zip(trajs, old, vllm_old):
                m = self._loss_mask(t) > 0
                if m.any():
                    gaps.append(np.abs(o[m] - v[m]).mean())
            stats["lp_gap_hf_vllm"] = float(np.mean(gaps)) if gaps else 0.0
        else:
            old = vllm_old
            stats["lp_gap_hf_vllm"] = float("nan")
        stats["prep_s"] = time.perf_counter() - t0

        # optimisation
        rng = np.random.default_rng(cfg.seed + self.optimizer_steps)
        mb_size = cfg.minibatch_size or B
        micro = cfg.hf_micro_batch_size
        agg_stats: dict[str, list[float]] = {}
        grad_norms, ent_vals, lrs = [], [], []
        n_steps = 0
        for epoch in range(cfg.ppo_epochs):
            perm = rng.permutation(B)
            for ms in range(0, B, mb_size):
                mb_idx = perm[ms:ms + mb_size].tolist()
                mb_idx.sort(key=lambda i: len(trajs[i].full_ids))          # less padding per micro-batch
                if cfg.loss_agg == "seq-mean-token-mean":
                    normaliser = float(len(mb_idx))
                else:
                    normaliser = float(sum(self._loss_mask(trajs[i]).sum() for i in mb_idx)) or 1.0
                self.policy.optimizer.zero_grad(set_to_none=True)
                for us in range(0, len(mb_idx), micro):
                    idx = mb_idx[us:us + micro]
                    sub = [trajs[i] for i in idx]
                    batch = self._collate(sub, {"old": [old[i] for i in idx], "ref": [ref[i] for i in idx]})
                    new_lp, ent = self.policy.logprobs(batch["input_ids"], batch["attention_mask"], with_grad=True,
                                                       use_adapter=True, want_entropy=True)
                    loss, st = grpo_loss(new_lp, batch["old"], batch["ref"], batch["adv"], batch["mask"],
                                         clip_eps=cfg.clip_eps, kl_coef=cfg.kl_coef, kl_type=cfg.kl_type,
                                         agg=cfg.loss_agg, normaliser=normaliser)
                    loss.backward()
                    for k, v in st.items():
                        agg_stats.setdefault(k, []).append(v)
                    agg_stats.setdefault("loss", []).append(float(loss.detach()) * normaliser / max(st["n_tokens"], 1.0))
                    with torch.no_grad():
                        m = batch["mask"]
                        ent_vals.append(float((ent * m).sum() / m.sum().clamp(min=1.0)))
                    del batch, new_lp, ent, loss
                gn = torch.nn.utils.clip_grad_norm_([p for p in self.policy.model.parameters() if p.requires_grad], cfg.max_grad_norm)
                grad_norms.append(float(gn))
                if not torch.isfinite(gn):
                    log.error("non-finite gradient norm; skipping this optimizer step")
                    self.policy.optimizer.zero_grad(set_to_none=True)
                    continue
                self.policy.optimizer.step()
                self.policy.scheduler.step()
                lrs.append(self.policy.scheduler.get_last_lr()[0])
                self.optimizer_steps += 1
                n_steps += 1
        self.policy.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        w = np.asarray(agg_stats.get("n_tokens", [1.0]))
        for k, vals in agg_stats.items():
            if k == "n_tokens":
                continue
            v = np.asarray(vals)
            stats[k] = float((v * w).sum() / w.sum()) if len(v) == len(w) and w.sum() > 0 else float(v.mean())
        stats["entropy"] = float(np.average(ent_vals, weights=w)) if ent_vals else float("nan")
        stats["grad_norm"] = float(np.mean(grad_norms)) if grad_norms else float("nan")
        stats["grad_norm_max"] = float(np.max(grad_norms)) if grad_norms else float("nan")
        stats["lr"] = float(lrs[-1]) if lrs else float(self.policy.scheduler.get_last_lr()[0])
        stats["optimizer_steps"] = n_steps
        stats["n_tokens_trained"] = float(w.sum())
        stats["train_s"] = time.perf_counter() - t0
        return stats
