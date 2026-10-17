"""The trainable side: Qwen3-8B loaded with bitsandbytes NF4 + a LoRA adapter (QLoRA).

vLLM cannot back-propagate, so after each rollout phase the vLLM engine sleeps (weights to CPU
RAM) and this model takes the GPU: it computes per-token log-probs of the sampled trajectories
with gradients, the GRPO update (src/grpo/backprop.py) steps the LoRA weights, the adapter is
saved to disk for vLLM to pick up, and the model is parked again. Three parking modes:

  offload  model.to("cpu") / .to("cuda")   bnb 4-bit params move as packed uint8 + quant state
  reload   free the model, rebuild from disk + adapter next step (slow but always works)
  keep     stay on the GPU; only viable with a small VLLM_GPU_MEM

`swap_check()` verifies at start-up that offload reproduces the pre-swap logits and falls
back to reload if it does not.
"""
from __future__ import annotations

import gc
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.config import TrainConfig

log = logging.getLogger(__name__)


class LocalPolicy:
    def __init__(self, cfg: TrainConfig, adapter_path: str | None = None, total_optimizer_steps: int = 1):
        from transformers import AutoTokenizer
        self.cfg = cfg
        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        self.adapter_path = adapter_path
        self.total_optimizer_steps = max(1, total_optimizer_steps)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self._opt_state_cpu = None
        self._sched_state = None
        self.on_gpu = False
        self.swap_mode = cfg.hf_swap_mode
        self.build()

    # ---- construction -------------------------------------------------------------------
    def build(self) -> None:
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        t0 = time.time()
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(self.cfg.base_model, quantization_config=bnb, dtype=torch.bfloat16,
                                                     device_map={"": 0}, attn_implementation="sdpa")
        model.config.use_cache = False
        # Done by hand instead of peft.prepare_model_for_kbit_training: that helper upcasts every non-quantised
        # parameter to fp32, which on Qwen3-8B means the 622M-parameter lm_head and therefore fp32 logits
        # (7 GB for 2 x 6k tokens) and ~2.5 GB more resident weights. bf16 norms/embeddings are fine for LoRA.
        for p in model.parameters():
            p.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        if self.adapter_path:
            model = PeftModel.from_pretrained(model, self.adapter_path, is_trainable=True)
        else:
            lcfg = LoraConfig(r=self.cfg.lora_r, lora_alpha=self.cfg.lora_alpha, lora_dropout=self.cfg.lora_dropout,
                              target_modules=self.cfg.lora_target_modules, task_type="CAUSAL_LM", bias="none")
            model = get_peft_model(model, lcfg)
        self.model = model
        self.on_gpu = True
        params = [p for p in model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in params)
        self.optimizer = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=(0.9, 0.999), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_lambda)
        if self._opt_state_cpu is not None:
            self.optimizer.load_state_dict(self._opt_state_cpu)
            self._opt_state_cpu = None
        if self._sched_state is not None:
            self.scheduler.load_state_dict(self._sched_state)
            self._sched_state = None
        log.info("HF QLoRA policy built in %.0fs: %.1fM trainable params, GPU alloc %.1f GB",
                 time.time() - t0, n_train / 1e6, torch.cuda.memory_allocated() / 1e9)

    def _lr_lambda(self, step: int) -> float:
        w = self.cfg.lr_warmup_steps
        if step < w:
            return (step + 1) / w
        if self.cfg.lr_schedule == "cosine":
            p = min(1.0, (step - w) / max(1, self.total_optimizer_steps - w))
            return 0.5 * (1 + math.cos(math.pi * p))
        return 1.0

    # ---- GPU residency --------------------------------------------------------------------
    def offload(self) -> None:
        if not self.on_gpu or self.swap_mode == "keep":
            return
        t0 = time.perf_counter()
        if self.swap_mode == "offload":
            self.model.to("cpu")
            _move_optimizer_state(self.optimizer, "cpu")
        else:                                       # reload
            self._opt_state_cpu = _state_dict_to(self.optimizer.state_dict(), "cpu")
            self._sched_state = self.scheduler.state_dict()
            self.model = self.optimizer = self.scheduler = None
        self.on_gpu = False
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log.debug("HF policy off GPU (%s) in %.1fs; GPU alloc %.2f GB", self.swap_mode, time.perf_counter() - t0,
                  torch.cuda.memory_allocated() / 1e9)

    def restore(self) -> None:
        if self.on_gpu:
            return
        t0 = time.perf_counter()
        if self.swap_mode == "offload":
            self.model.to(self.device)
            _move_optimizer_state(self.optimizer, self.device)
            self.on_gpu = True
        else:
            self.build()
        log.debug("HF policy on GPU in %.1fs; GPU alloc %.2f GB", time.perf_counter() - t0, torch.cuda.memory_allocated() / 1e9)

    def swap_check(self) -> bool:
        """Round-trip the model through CPU and compare logits. On failure switch to reload mode."""
        if self.swap_mode != "offload":
            return True
        ids = torch.tensor([self.tokenizer.encode("The capital of France is", add_special_tokens=False)], device=self.device)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            before = self.model(input_ids=ids).logits[0, -1].float()
        self.offload()
        self.restore()
        with torch.no_grad():
            after = self.model(input_ids=ids).logits[0, -1].float()
        self.model.train(was_training)
        ok = bool(torch.allclose(before, after, atol=1e-2, rtol=1e-2))
        if not ok:
            log.warning("offload round-trip changed the logits (max |diff| %.3g): falling back to HF_SWAP_MODE=reload",
                        (before - after).abs().max().item())
            self.swap_mode = "reload"
        else:
            log.info("offload round-trip ok (max |diff| %.3g)", (before - after).abs().max().item())
        return ok

    # ---- log-probs ------------------------------------------------------------------------
    def logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, with_grad: bool, use_adapter: bool = True,
                 want_entropy: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Per-token log p(x_t | x_<t) for t = 1..T-1, shape (B, T-1) fp32, aligned with input_ids[:, 1:].
        use_adapter=False gives the reference policy (adapters disabled). Entropy is detached."""
        if not self.on_gpu:
            raise RuntimeError("HF policy is not on the GPU; call restore() first")
        if with_grad:
            self.model.train()      # HF applies gradient checkpointing only in training mode; eval() would store all activations
        # The decoder runs once; lm_head + log-softmax run per chunk of HF_LOGITS_CHUNK positions inside an
        # activation checkpoint, so the (B, T, 152k) logits never exist as one tensor (bf16 alone would be
        # 3.6 GB for 2 x 6k tokens, and autograd would allocate full-size gradient buffers for every slice).
        base = self.model.get_base_model()
        decoder, lm_head = base.model, base.lm_head
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            if use_adapter:
                out = decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            else:
                with self.model.disable_adapter():
                    out = decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            hidden = out.last_hidden_state[:, :-1]
            labels = input_ids[:, 1:]
            lp_chunks, ent_chunks = [], []
            step = self.cfg.hf_logits_chunk
            for s in range(0, hidden.shape[1], step):
                h, lb = hidden[:, s:s + step], labels[:, s:s + step]
                if with_grad:
                    lp, ent = checkpoint(_chunk_logprobs, h, lb, lm_head, want_entropy, use_reentrant=False)
                else:
                    lp, ent = _chunk_logprobs(h, lb, lm_head, want_entropy)
                lp_chunks.append(lp)
                if want_entropy:
                    ent_chunks.append(ent)
            lp = torch.cat(lp_chunks, dim=1)
            ent = torch.cat(ent_chunks, dim=1).detach() if want_entropy else None
        return lp, ent

    # ---- persistence ----------------------------------------------------------------------
    def save_adapter(self, path: Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.adapter_path = str(path)
        return path

    def save_optimizer(self, path: Path) -> None:
        torch.save({"optimizer": _state_dict_to(self.optimizer.state_dict(), "cpu"), "scheduler": self.scheduler.state_dict()}, path)

    def load_optimizer(self, path: Path) -> None:
        st = torch.load(path, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(st["optimizer"])
        self.scheduler.load_state_dict(st["scheduler"])
        _move_optimizer_state(self.optimizer, self.device if self.on_gpu else "cpu")

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()


def _chunk_logprobs(hidden: torch.Tensor, labels: torch.Tensor, lm_head: torch.nn.Module, want_entropy: bool):
    """lm_head + fp32 log-softmax for a (B, n, d) chunk of hidden states; recomputed in backward
    (checkpoint) so only the small hidden-state chunk stays resident."""
    lg = lm_head(hidden).float()
    lse = torch.logsumexp(lg, dim=-1)
    lp = lg.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - lse
    if want_entropy:
        with torch.no_grad():
            p = torch.softmax(lg, dim=-1)
            ent = lse - (p * lg).sum(-1)
    else:
        ent = torch.zeros((), device=lg.device)
    return lp, ent


def _move_optimizer_state(opt: torch.optim.Optimizer, device) -> None:
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v) and v.dim() > 0:      # leave the scalar step counters where they are
                state[k] = v.to(device)


def _state_dict_to(sd: dict, device) -> dict:
    def mv(x):
        if torch.is_tensor(x):
            return x.to(device)
        if isinstance(x, dict):
            return {k: mv(v) for k, v in x.items()}
        if isinstance(x, list):
            return [mv(v) for v in x]
        return x
    return mv(sd)
