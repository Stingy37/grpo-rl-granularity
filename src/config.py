"""Single source of truth for every knob in the harness.

Every field of TrainConfig can be set with an environment variable of the same name in
upper case (scripts/train_rl.sh does exactly that); main.parse_args() builds the config
with TrainConfig.from_env(). Paths default to ~/datasets, where the wiki-18 index, the
flat e5 vectors, the passage parquet, HotpotQA and the model weights live.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = Path(os.environ.get("DATASETS_DIR", Path.home() / "datasets"))

# wiki-18 corpus constants (PeterJinGo/wiki-18-corpus, same rows as e5_Flat.index)
N_PASSAGES = 21_015_324
DIM = 768
# IndexFlatIP on disk = fourcc(4) d(4) ntotal(8) dummy(16) is_trained(1) metric(4) codes-len(8) = 45 bytes,
# then raw float32 rows. The whole 64.5 GB file is never loaded: rows are pread() on demand.
FLAT_HEADER_BYTES = 45
PASSAGE_ROW_GROUP = 8192


@dataclass
class TrainConfig:
    # ---- paths -----------------------------------------------------------------------------
    output_dir: Path = PROJECT_DIR / "output" / "rl"
    base_model: str = str(DATASETS_DIR / "models" / "Qwen3-8B")
    e5_dir: str = str(DATASETS_DIR / "models" / "e5-base-v2")
    index_path: Path = DATASETS_DIR / "wiki-18" / "compressed" / "wiki18_e5_opq64_ivf16384_pq64.index"
    flat_index_path: Path = DATASETS_DIR / "wiki-18" / "raw" / "wiki-18-e5-index" / "e5_Flat.index"
    passages_path: Path = DATASETS_DIR / "wiki-18" / "compressed" / "wiki18_passages.parquet"
    # a parquet/jsonl file, or a directory whose train-*.parquet shards are all read (HF hotpotqa/hotpot_qa layout)
    hotpot_train_path: Path = DATASETS_DIR / "hotpotqa" / "hf" / "distractor"
    hotpot_dev_path: Path = DATASETS_DIR / "hotpotqa" / "hf" / "distractor" / "validation-00000-of-00001.parquet"
    resume_from: str = ""            # adapter dir written by a previous run (output/rl/adapters/step_N)

    # ---- retrieval (dataset/, src/retriever/) ----------------------------------------------
    nprobe: int = 128                # IVF lists scanned per query (coarse step, centroids in RAM)
    rerank_candidates: int = 100     # top-R PQ candidates whose exact fp32 rows are pread() from the flat file
    retrieve_topk: int = 10          # exact-reranked passages returned per query
    context_topk: int = 5            # how many of those go into <information> (Search-R1 uses 3)
    max_obs_tokens: int = 1024       # hard cap on one <information> block (Search-R1: max_obs_length=500 for 3 docs)
    faiss_threads: int = 16
    rerank_threads: int = 32         # concurrent 3 KB NFS reads for the exact re-rank
    e5_device: str = "cuda"          # the 110M-param query encoder; fp16 on GPU costs ~0.3 GB

    # ---- rollout (src/engine/rollout.py) ---------------------------------------------------
    num_groups: int = 8              # HotpotQA questions per GRPO step
    rollouts_per_group: int = 8      # G in GRPO
    max_tokens: int = 8192           # whole response: think block (generated + inserted) + answer
    think_fraction: float = 0.75     # share of MAX_TOKENS the think block may use (6144); the rest is for the answer
    max_model_len: int = 9216        # vLLM context: prompt (~150-250 tokens) + MAX_TOKENS must fit
    temperature: float = 1.0         # rollout sampling (GRPO exploration)
    top_p: float = 1.0
    eval_temperature: float = 0.6    # held-out eval: Qwen3's recommended thinking-mode sampling (greedy loops)
    eval_top_p: float = 0.95
    eval_top_k: int = 20
    rollout_seed_per_step: bool = True

    # ---- vLLM (src/engine/vllm.py) ---------------------------------------------------------
    vllm_gpu_mem: float = 0.80       # fraction of the 33 GB MIG slice vLLM may own while awake
    vllm_quantization: str = "fp8"
    vllm_kv_cache_dtype: str = "fp8"
    vllm_max_num_seqs: int = 128
    vllm_enforce_eager: bool = False  # True skips CUDA-graph capture: faster start, slower decode
    vllm_sleep_level: int = 1        # 1 = weights parked in CPU RAM, 2 = weights discarded and reloaded

    # ---- LoRA / optimiser (src/engine/local_llm.py) ----------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lr: float = 2e-5                 # LoRA needs ~10x Search-R1's full-FT 1e-6
    lr_warmup_steps: int = 10
    lr_schedule: str = "constant"    # constant | cosine
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    hf_swap_mode: str = "offload"    # offload: model.to(cpu) between steps | reload: rebuild from disk | keep: stay on GPU
    hf_micro_batch_size: int = 2     # sequences per forward/backward
    hf_logits_chunk: int = 1024      # tokens per log-softmax chunk (bounds the fp32 logits working set)

    # ---- GRPO (src/grpo/) ------------------------------------------------------------------
    ppo_epochs: int = 1              # gradient passes over one rollout batch (multi-step reuse)
    minibatch_size: int = 0          # sequences per optimiser step; 0 = the whole batch
    clip_eps: float = 0.2
    kl_coef: float = 0.001           # Search-R1: 0.001, low_var_kl
    kl_type: str = "low_var_kl"      # low_var_kl (k3) | kl | mse
    adv_norm_std: bool = True        # divide group-centred rewards by group std (False = Dr. GRPO)
    loss_agg: str = "token-mean"     # token-mean | seq-mean-token-mean
    # hf: the NF4 model with adapters disabled (same numerics as the policy, KL measures only the LoRA's drift)
    # vllm: the fp8 model without adapter via prompt_logprobs (free, but fp8-vs-NF4 noise alone reads as KL ~0.2)
    ref_logprobs_source: str = "hf"
    old_logprobs_source: str = "hf"    # hf: extra no-grad pass before the update | vllm: rollout logprobs

    # ---- reward (src/grpo/reward.py, from the toy train_grpo.py) ---------------------------
    w_under: float = 0.5
    w_over: float = 0.1
    under_lo: float = 0.3
    under_hi: float = 0.7
    over_lo: float = 0.5
    over_hi: float = 1.0
    pen_power: int = 2
    q_gate: float = 0.7
    format_penalty: float = 0.0      # optional extra penalty when no <answer> is produced (0 = off)

    # ---- judge (src/grpo/judge.py) ---------------------------------------------------------
    judge_use_gold: bool = False     # also score factual accuracy against the HotpotQA gold answer
    judge_max_tokens: int = 16
    judge_max_chars: int = 12000     # trajectory text shown to the judge is clipped to this

    # ---- schedule / bookkeeping (src/train_harness/) --------------------------------------
    total_steps: int = 300
    eval_every: int = 20
    eval_n: int = 100                # held-out HotpotQA questions per evaluation
    save_every: int = 20
    keep_last: int = 3               # step checkpoints kept besides best/
    best_metric: str = "em"          # em | f1 | reward
    sample_dump_every: int = 5       # write a few full trajectories to logs/samples every N steps
    seed: int = 0
    log_level: str = "INFO"

    # ---- derived -------------------------------------------------------------------------
    @property
    def batch_size(self) -> int:
        return self.num_groups * self.rollouts_per_group

    @property
    def lora_target_modules(self) -> list[str]:
        return [t.strip() for t in self.lora_targets.split(",") if t.strip()]

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def ckpt_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def adapter_dir(self) -> Path:
        return self.output_dir / "adapters"

    # ---- construction --------------------------------------------------------------------
    @classmethod
    def from_env(cls, env: dict | None = None) -> "TrainConfig":
        env = os.environ if env is None else env
        kwargs = {}
        for f in dataclasses.fields(cls):
            raw = env.get(f.name.upper())
            if raw is None or raw == "":
                continue
            kwargs[f.name] = _coerce(raw, f)
        return cls(**kwargs)

    def validate(self) -> None:
        for p in (self.index_path, self.passages_path, self.hotpot_dev_path):
            if not Path(p).exists():
                raise FileNotFoundError(p)
        if self.rerank_candidates and not Path(self.flat_index_path).exists():
            raise FileNotFoundError(f"{self.flat_index_path} (set RERANK_CANDIDATES=0 to skip exact re-ranking)")
        if not Path(self.base_model).exists():
            raise FileNotFoundError(self.base_model)
        if self.context_topk > self.retrieve_topk:
            raise ValueError("CONTEXT_TOPK must be <= RETRIEVE_TOPK")
        if not 0.1 <= self.think_fraction <= 0.95:
            raise ValueError("THINK_FRACTION must be in [0.1, 0.95]")
        if self.max_tokens + 512 > self.max_model_len:
            raise ValueError("MAX_TOKENS + 512 (prompt headroom) must be <= MAX_MODEL_LEN")
        if self.max_obs_tokens >= int(self.max_tokens * self.think_fraction):
            raise ValueError("MAX_OBS_TOKENS must be smaller than the thinking budget THINK_FRACTION * MAX_TOKENS")
        if self.rollouts_per_group < 2:
            raise ValueError("ROLLOUTS_PER_GROUP must be >= 2 for a group baseline")
        if self.minibatch_size and self.minibatch_size % self.hf_micro_batch_size:
            raise ValueError("MINIBATCH_SIZE must be a multiple of HF_MICRO_BATCH_SIZE")
        if self.kl_type not in ("low_var_kl", "kl", "mse"):
            raise ValueError(f"KL_TYPE {self.kl_type!r}")
        if self.hf_swap_mode not in ("offload", "reload", "keep"):
            raise ValueError(f"HF_SWAP_MODE {self.hf_swap_mode!r}")
        if self.best_metric not in ("em", "f1", "reward"):
            raise ValueError(f"BEST_METRIC {self.best_metric!r}")

    def ensure_dirs(self) -> None:
        for d in (self.output_dir, self.log_dir, self.ckpt_dir, self.adapter_dir, self.log_dir / "samples"):
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(self).items()}

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))


def _coerce(raw: str, f: dataclasses.Field):
    t = f.type if not isinstance(f.type, str) else f.type
    default = f.default
    if isinstance(default, bool) or t in (bool, "bool"):
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")
    if isinstance(default, int) and not isinstance(default, bool) or t in (int, "int"):
        return int(raw)
    if isinstance(default, float) or t in (float, "float"):
        return float(raw)
    if isinstance(default, Path) or t in (Path, "Path"):
        return Path(raw).expanduser()
    return raw
