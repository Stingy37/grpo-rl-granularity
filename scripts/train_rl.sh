#!/usr/bin/env bash
# GRPO + Search-R1-style retrieval training of Qwen3-8B (fp8 vLLM rollouts, QLoRA updates).
# Every hyper-parameter is an environment variable read by src/config.py; the values below are
# defaults that an exported variable overrides, e.g.  NUM_GROUPS=16 TOTAL_STEPS=500 bash scripts/train_rl.sh
# Extra CLI flags are passed through to src/main.py (--smoke, --print-config, --check-retrieval "q").
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/.conda/envs/grpo-rl/bin/python}"      # created by scripts/setup_env.sh

# ---- data / models (defaults point at ~/datasets) --------------------------------------------
export DATASETS_DIR="${DATASETS_DIR:-$HOME/datasets}"
export BASE_MODEL="${BASE_MODEL:-$DATASETS_DIR/models/Qwen3-8B}"
export E5_DIR="${E5_DIR:-$DATASETS_DIR/models/e5-base-v2}"
export INDEX_PATH="${INDEX_PATH:-$DATASETS_DIR/wiki-18/compressed/wiki18_e5_opq64_ivf16384_pq64.index}"
export FLAT_INDEX_PATH="${FLAT_INDEX_PATH:-$DATASETS_DIR/wiki-18/raw/wiki-18-e5-index/e5_Flat.index}"
export PASSAGES_PATH="${PASSAGES_PATH:-$DATASETS_DIR/wiki-18/compressed/wiki18_passages.parquet}"
export HOTPOT_TRAIN_PATH="${HOTPOT_TRAIN_PATH:-$DATASETS_DIR/hotpotqa/hf/distractor}"   # dir of train-*.parquet
export HOTPOT_DEV_PATH="${HOTPOT_DEV_PATH:-$DATASETS_DIR/hotpotqa/hf/distractor/validation-00000-of-00001.parquet}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output/rl}"
export RESUME_FROM="${RESUME_FROM:-}"                  # output/rl/adapters/step_N to continue a run

# ---- retrieval: IVF coarse (RAM) -> OPQ/PQ scan (paged lists) -> exact re-rank from the flat file ----
export NPROBE="${NPROBE:-128}"
export RERANK_CANDIDATES="${RERANK_CANDIDATES:-100}"
export RETRIEVE_TOPK="${RETRIEVE_TOPK:-10}"
export CONTEXT_TOPK="${CONTEXT_TOPK:-5}"               # passages placed in <information> (Search-R1: 3)
export MAX_OBS_TOKENS="${MAX_OBS_TOKENS:-1024}"       # cap on one <information> block, so a single retrieval cannot flood the thinking budget
export FAISS_THREADS="${FAISS_THREADS:-16}"
export RERANK_THREADS="${RERANK_THREADS:-32}"

# ---- GRPO batch: NUM_GROUPS questions x ROLLOUTS_PER_GROUP rollouts per step ---------------
export NUM_GROUPS="${NUM_GROUPS:-8}"
export ROLLOUTS_PER_GROUP="${ROLLOUTS_PER_GROUP:-8}"
# Native Qwen3 thinking is the reasoning channel; <search>/<information> live inside the think block.
# No turn cap: the response gets MAX_TOKENS, THINK_FRACTION of it for thinking (generated + inserted
# information), the rest for the answer. Thinking that hits its budget is force-closed with </think>.
export MAX_TOKENS="${MAX_TOKENS:-8192}"               # 6144 thinking + 2048 answer at THINK_FRACTION 0.75
export THINK_FRACTION="${THINK_FRACTION:-0.75}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"          # must be >= MAX_TOKENS + 512 (prompt headroom)
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"     # Qwen3 thinking-mode recommendation (greedy tends to loop)
export EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
export EVAL_TOP_K="${EVAL_TOP_K:-20}"

# ---- vLLM (rollouts, judge, reference log-probs) ---------------------------------------------
export VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.80}"
export VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-fp8}"
export VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-fp8}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
export VLLM_SLEEP_LEVEL="${VLLM_SLEEP_LEVEL:-1}"

# ---- LoRA / optimiser (HF QLoRA side) --------------------------------------------------------
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
export LORA_TARGETS="${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
export LR="${LR:-2e-5}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"
export LR_SCHEDULE="${LR_SCHEDULE:-constant}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
export HF_SWAP_MODE="${HF_SWAP_MODE:-offload}"         # offload | reload | keep
export HF_MICRO_BATCH_SIZE="${HF_MICRO_BATCH_SIZE:-2}"
export HF_LOGITS_CHUNK="${HF_LOGITS_CHUNK:-1024}"

# ---- GRPO objective -------------------------------------------------------------------------
export PPO_EPOCHS="${PPO_EPOCHS:-1}"                   # gradient passes over each rollout batch (multi-step reuse)
export MINIBATCH_SIZE="${MINIBATCH_SIZE:-0}"           # sequences per optimizer step; 0 = whole batch
export CLIP_EPS="${CLIP_EPS:-0.2}"
export KL_COEF="${KL_COEF:-0.001}"
export KL_TYPE="${KL_TYPE:-low_var_kl}"
export ADV_NORM_STD="${ADV_NORM_STD:-1}"
export LOSS_AGG="${LOSS_AGG:-token-mean}"
export REF_LOGPROBS_SOURCE="${REF_LOGPROBS_SOURCE:-hf}"   # hf: NF4 model, adapters off | vllm: fp8 model (KL then includes fp8-vs-NF4 noise ~0.2)
export OLD_LOGPROBS_SOURCE="${OLD_LOGPROBS_SOURCE:-hf}"

# ---- reward (toy train_grpo.py) and judge ----------------------------------------------------
export W_UNDER="${W_UNDER:-0.5}"
export W_OVER="${W_OVER:-0.1}"
export UNDER_LO="${UNDER_LO:-0.3}"
export UNDER_HI="${UNDER_HI:-0.7}"
export OVER_LO="${OVER_LO:-0.5}"
export OVER_HI="${OVER_HI:-1.0}"
export PEN_POWER="${PEN_POWER:-2}"
export Q_GATE="${Q_GATE:-0.7}"
export FORMAT_PENALTY="${FORMAT_PENALTY:-0.0}"
export JUDGE_USE_GOLD="${JUDGE_USE_GOLD:-0}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-16}"

# ---- schedule / checkpoints -----------------------------------------------------------------
export TOTAL_STEPS="${TOTAL_STEPS:-300}"
export EVAL_EVERY="${EVAL_EVERY:-20}"
export EVAL_N="${EVAL_N:-100}"
export SAVE_EVERY="${SAVE_EVERY:-20}"
export KEEP_LAST="${KEEP_LAST:-3}"
export BEST_METRIC="${BEST_METRIC:-em}"
export SAMPLE_DUMP_EVERY="${SAMPLE_DUMP_EVERY:-5}"
export SEED="${SEED:-0}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

# ---- process environment --------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"            # all weights are local
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"   # its kernel is JIT-built with nvcc, absent on this pod
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"   # less fragmentation with variable-length batches
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_DIR/logs"
cd "$PROJECT_DIR"
exec "$PY" -m src.main "$@" 2>&1 | tee -a "$OUTPUT_DIR/logs/console_$(date +%Y%m%d_%H%M%S).log"
