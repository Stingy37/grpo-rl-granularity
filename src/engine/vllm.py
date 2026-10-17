"""Everything that touches vLLM.

One vLLM instance serves three roles, all on the same fp8-quantised Qwen3-8B:
  * policy rollouts         base + the current LoRA adapter (a LoRARequest pointing at the
                            adapter directory the HF side saved after the last update)
  * reference model (KL)    base without adapter: prompt_logprobs over a finished trajectory
  * judge                   base without adapter, chat template with thinking disabled

Between the rollout phase and the gradient step the engine goes to sleep: level 1 parks the
weights in CPU RAM and frees the KV cache so the HF QLoRA model can own the GPU; wake_up()
brings them back. The MIG slice has ~33 GB, so only one of the two models is ever resident.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from src.config import TrainConfig

log = logging.getLogger(__name__)


@dataclass
class GenOutput:
    token_ids: list[int]
    logprobs: np.ndarray            # logprob of each generated token under the sampling model
    text: str
    finish_reason: str              # "stop" | "length" | ...
    stop_reason: object = None      # the stop string / token that ended it, if any


@dataclass
class EngineTimings:
    generate_s: float = 0.0
    prompt_logprobs_s: float = 0.0
    chat_s: float = 0.0
    sleep_s: float = 0.0
    wake_s: float = 0.0
    extra: dict = field(default_factory=dict)


class VLLMEngine:
    def __init__(self, cfg: TrainConfig):
        from vllm import LLM
        self.cfg = cfg
        self.timings = EngineTimings()
        t0 = time.time()
        self.llm = LLM(
            model=cfg.base_model,
            dtype="bfloat16",
            quantization=cfg.vllm_quantization or None,
            kv_cache_dtype=cfg.vllm_kv_cache_dtype,
            enable_lora=True,
            max_lora_rank=cfg.lora_r,
            max_loras=1,
            max_cpu_loras=2,
            gpu_memory_utilization=cfg.vllm_gpu_mem,
            max_model_len=cfg.max_model_len,
            max_num_seqs=cfg.vllm_max_num_seqs,
            enable_sleep_mode=True,
            enable_prefix_caching=True,
            enforce_eager=cfg.vllm_enforce_eager,
            seed=cfg.seed,
            trust_remote_code=False,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.lora_request = None
        self._lora_ids = 0
        self.asleep = False
        log.info("vLLM up in %.0fs: %s quant=%s kv=%s mem=%.2f max_len=%d", time.time() - t0, cfg.base_model,
                 cfg.vllm_quantization, cfg.vllm_kv_cache_dtype, cfg.vllm_gpu_mem, cfg.max_model_len)

    # ---- adapter --------------------------------------------------------------------------
    def set_adapter(self, path: str | None, tag: str = "") -> None:
        """Point rollouts at a new adapter directory. A fresh lora_int_id makes vLLM load the new
        weights instead of reusing the cached ones for the previous step."""
        if path is None:
            self.lora_request = None
            return
        from vllm.lora.request import LoRARequest
        self._lora_ids += 1
        self.lora_request = LoRARequest(lora_name=f"adapter-{tag or self._lora_ids}", lora_int_id=self._lora_ids, lora_path=str(path))

    # ---- sampling params ------------------------------------------------------------------
    def rollout_params(self, max_tokens: int, stop: list[str], seed: int | None = None):
        from vllm import SamplingParams
        return SamplingParams(
            n=1, temperature=self.cfg.temperature, top_p=self.cfg.top_p, max_tokens=max_tokens,
            stop=stop, include_stop_str_in_output=True, logprobs=0,
            skip_special_tokens=False, seed=seed,
        )

    def eval_params(self, max_tokens: int, stop: list[str], seed: int | None = None):
        """Held-out evaluation: Qwen3's recommended thinking-mode sampling (T 0.6, top-p 0.95, top-k 20)
        with a fixed seed per question; EVAL_TEMPERATURE=0 gives greedy decoding."""
        from vllm import SamplingParams
        return SamplingParams(
            n=1, temperature=self.cfg.eval_temperature, top_p=self.cfg.eval_top_p, top_k=self.cfg.eval_top_k,
            max_tokens=max_tokens, stop=stop, include_stop_str_in_output=True, logprobs=0,
            skip_special_tokens=False, seed=seed,
        )

    # ---- generation -----------------------------------------------------------------------
    def generate(self, token_id_lists: list[list[int]], sampling_params, use_lora: bool = True) -> list[GenOutput]:
        """Continue each token sequence. `sampling_params` is one SamplingParams or one per request."""
        self._assert_awake()
        if not token_id_lists:
            return []
        prompts = [{"prompt_token_ids": list(ids)} for ids in token_id_lists]
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, sampling_params, use_tqdm=False,
                                 lora_request=self.lora_request if use_lora else None)
        self.timings.generate_s += time.perf_counter() - t0
        res = []
        for o in outs:
            c = o.outputs[0]
            ids = list(c.token_ids)
            lps = np.zeros(len(ids), dtype=np.float32)
            if c.logprobs is not None:
                for j, (tid, d) in enumerate(zip(ids, c.logprobs)):
                    if d and tid in d:
                        lps[j] = d[tid].logprob
                    elif d:                                   # should not happen with logprobs=0; take the recorded entry
                        lps[j] = next(iter(d.values())).logprob
            res.append(GenOutput(token_ids=ids, logprobs=lps, text=c.text, finish_reason=c.finish_reason, stop_reason=c.stop_reason))
        return res

    def prompt_logprobs(self, token_id_lists: list[list[int]], use_lora: bool = False) -> list[np.ndarray]:
        """Log p(token_i | tokens_<i) for every position of each sequence (index 0 is NaN), under the
        base model (use_lora=False: the reference policy) or the current adapter."""
        from vllm import SamplingParams
        self._assert_awake()
        if not token_id_lists:
            return []
        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0, logprobs=None, skip_special_tokens=False)
        prompts = [{"prompt_token_ids": list(ids)} for ids in token_id_lists]
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, sp, use_tqdm=False, lora_request=self.lora_request if use_lora else None)
        self.timings.prompt_logprobs_s += time.perf_counter() - t0
        res = []
        for ids, o in zip(token_id_lists, outs):
            lp = np.full(len(ids), np.nan, dtype=np.float32)
            pl = o.prompt_logprobs or []
            for i, d in enumerate(pl):
                if d is None or i >= len(ids):
                    continue
                tid = ids[i]
                if tid in d:
                    lp[i] = d[tid].logprob
                elif d:
                    lp[i] = next(iter(d.values())).logprob
            res.append(lp)
        return res

    def chat(self, conversations: list[list[dict]], max_tokens: int, temperature: float = 0.0, use_lora: bool = False) -> list[str]:
        """Plain chat completions (the judge). Thinking is disabled through the chat template."""
        from vllm import SamplingParams
        self._assert_awake()
        if not conversations:
            return []
        sp = SamplingParams(n=1, temperature=temperature, max_tokens=max_tokens)
        t0 = time.perf_counter()
        try:
            outs = self.llm.chat(conversations, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False},
                                 lora_request=self.lora_request if use_lora else None)
        except TypeError:   # tokenizer without the enable_thinking kwarg
            outs = self.llm.chat(conversations, sp, use_tqdm=False, lora_request=self.lora_request if use_lora else None)
        self.timings.chat_s += time.perf_counter() - t0
        return [o.outputs[0].text for o in outs]

    # ---- GPU residency --------------------------------------------------------------------
    def sleep(self) -> None:
        if self.asleep:
            return
        t0 = time.perf_counter()
        self.llm.sleep(level=self.cfg.vllm_sleep_level)
        self.asleep = True
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.timings.sleep_s += time.perf_counter() - t0
        log.debug("vLLM asleep (level %d); free GPU %.1f GB", self.cfg.vllm_sleep_level, free_gpu_gb())

    def wake(self) -> None:
        if not self.asleep:
            return
        t0 = time.perf_counter()
        self.llm.wake_up()
        self.asleep = False
        self.timings.wake_s += time.perf_counter() - t0
        log.debug("vLLM awake; free GPU %.1f GB", free_gpu_gb())

    def _assert_awake(self) -> None:
        if self.asleep:
            raise RuntimeError("vLLM engine is asleep; call wake() first")

    def reset_timings(self) -> EngineTimings:
        t, self.timings = self.timings, EngineTimings()
        return t


def free_gpu_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / 1e9
