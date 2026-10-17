# grpo-rl

GRPO training of a Search-R1-style retrieval agent (Qwen3-8B + LoRA) on HotpotQA with the
wiki-18 corpus, on one 33 GB MIG slice of an H200. vLLM serves the fp8 policy for rollouts,
the judge and the reference model; a bitsandbytes-NF4 copy of the model takes the GPU only
for the gradient step (QLoRA). The 64 GB wiki-18 e5 index is never loaded: retrieval runs on
a 1.5 GB OPQ64,IVF16384,PQ64 index (54 MB resident) with an exact re-rank that `pread()`s the
candidate rows from the flat file.

## Configuration reference

Every knob is an environment variable that `scripts/train_rl.sh` exports with the default shown; set it on the
command line to override (`NUM_GROUPS=16 LR=1e-5 bash scripts/train_rl.sh`). `src/config.py` reads each one under its
upper-case field name, and `bash scripts/train_rl.sh --print-config` shows the resolved values. The defaults here are
`train_rl.sh`'s. Running `python -m src.main` directly uses `config.py`'s instead, which differ only in
`MAX_TOKENS` (4096) and `MAX_MODEL_LEN` (8192). Rows marked *config.py only* are not in `train_rl.sh` but can still be
set the same way. Settings that shape what the run learns, or whether it fits on the GPU, also say **why** they have
their default.

### Command-line flags

`train_rl.sh` passes these through to `src/main.py`.

| Flag | What it does |
|---|---|
| `--smoke` | Tiny end-to-end run to check the pipeline: 2 groups × 2 rollouts, 2 steps, `MAX_TOKENS=1536`, eval on 4 questions after every step, optimizer state and samples saved every step. Writes to `output/rl_smoke/`, except the console log, which `train_rl.sh` still tees into `$OUTPUT_DIR/logs/` because the shell doesn't see the flag. Its numbers mean little: each rate moves in 25 % steps, and the 1152-token thinking budget is nearly always cut off. **Delete `output/rl_smoke/` before rerunning it.** Like any run it resumes from its `state.json`, and a finished smoke run has no steps left. |
| `--print-config` | Print the resolved configuration as JSON and exit. |
| `--check-retrieval "q"` | Run the retriever on one query, print the top `RETRIEVE_TOPK` passages and per-stage timings, and exit. Loads no LLM. |
| `--no-eval` | Same as `EVAL_EVERY=0`. |

### Paths and resuming

| Variable | Default | What it does |
|---|---|---|
| `PY` | `~/.conda/envs/grpo-rl/bin/python` | Interpreter: the env that `scripts/setup_env.sh` builds. |
| `DATASETS_DIR` | `~/datasets` | Root that the data and model defaults below are built from. |
| `BASE_MODEL` | `…/models/Qwen3-8B` | The one checkpoint behind everything: vLLM's fp8 policy, the judge, the reference model and the NF4 training copy. |
| `E5_DIR` | `…/models/e5-base-v2` | Query encoder. It must be the model the corpus vectors were made with (Search-R1's e5-base-v2), or queries land in a different embedding space. |
| `INDEX_PATH` | `…/wiki18_e5_opq64_ivf16384_pq64.index` | The 1.5 GB compressed index that is searched first (mmap, 54 MB resident). |
| `FLAT_INDEX_PATH` | `…/e5_Flat.index` | The 64.5 GB file of exact vectors. Only the `RERANK_CANDIDATES` rows per query are read from it. Required unless `RERANK_CANDIDATES=0`. |
| `PASSAGES_PATH` | `…/wiki18_passages.parquet` | Passage titles and text, looked up by id. |
| `HOTPOT_TRAIN_PATH` | `…/hotpotqa/hf/distractor` | Training questions: a directory of `train-*.parquet` shards, or one parquet / json / jsonl file. If it's missing, the dev split is shuffled and divided into held-out and train. |
| `HOTPOT_DEV_PATH` | `…/validation-00000-of-00001.parquet` | Dev split. The `EVAL_N` held-out questions are drawn from it once, with `SEED`. |
| `OUTPUT_DIR` | `output/rl` | Logs, adapters, checkpoints and `state.json`. **Rerunning with the same `OUTPUT_DIR` resumes automatically** from `state.json` (step, adapter, position in the question order). Use a new directory for a fresh run. |
| `RESUME_FROM` | empty | Adapter directory to continue from, e.g. `output/rl/adapters/step_00100`. It overrides the adapter named in `state.json`; without a `state.json` the step is read from the directory name. AdamW's state comes back only if the directory has `optimizer.pt` (written every `SAVE_EVERY` steps); otherwise the optimizer starts fresh. |

### Retrieval

| Variable | Default | What it does |
|---|---|---|
| `NPROBE` | 128 | IVF lists scanned per query, out of 16,384. **Why 128:** together with the re-rank it gives recall@10 of 0.81 against exact search at about 20 ms per query. More lists give better recall but page in more data (about 12 MB of lists per query at 128). |
| `RERANK_CANDIDATES` | 100 | Top PQ candidates whose exact vectors are read from the flat file and re-scored. **Why:** PQ scores are approximate. Re-scoring 100 exact rows recovers most of the exact ranking (any-gold@10 0.77, vs 0.82 for a full exact search) for a few milliseconds of reads. `0` skips the re-rank and the flat file. |
| `RETRIEVE_TOPK` | 10 | Passages returned per search after the re-rank. Only the first `CONTEXT_TOPK` reach the model, but all 10 count toward the eval's `any-gold` / `all-gold`. |
| `CONTEXT_TOPK` | 5 | Passages spliced into `<information>`. **Why 5 rather than Search-R1's 3:** HotpotQA questions need two articles, and the extra passages help find the second one. 10 would cost about 1.3k tokens of the thinking budget per search. Must be ≤ `RETRIEVE_TOPK`. |
| `MAX_OBS_TOKENS` | 1024 | Hard cap on one `<information>` block; passage text is trimmed to fit. **Why:** inserted tokens count against the thinking budget, so without a cap a single search could use most of it. Must be < `THINK_FRACTION × MAX_TOKENS`. |
| `FAISS_THREADS` | 16 | OpenMP threads for the coarse search and the PQ scan. At startup the retriever checks that batched search matches single-threaded search, and refuses to start if it doesn't (faiss 1.15.1 fails this). `FAISS_THREADS=1` is the escape hatch. |
| `RERANK_THREADS` | 32 | Concurrent `pread()`s of 3 KB rows from the flat file. **Why so many:** the reads are limited by NFS round-trip latency, not by CPU, so overlapping them is what makes the re-rank fast. |
| `E5_DEVICE` | `cuda` | *config.py only.* Where the 110M-parameter query encoder runs. On the GPU in fp16 it costs about 0.3 GB; `cpu` saves that but encodes more slowly. |

### Batch, token budgets and sampling

| Variable | Default | What it does |
|---|---|---|
| `NUM_GROUPS` | 8 | Questions per GRPO step. |
| `ROLLOUTS_PER_GROUP` | 8 | Rollouts per question (GRPO's G). The batch is `NUM_GROUPS × ROLLOUTS_PER_GROUP` = 64. **Why it matters:** a rollout's advantage is its reward relative to its own group, so a group teaches something only if its rollouts score differently. The smaller G is, the more groups come out identical (`zeroG`) and are wasted. Must be ≥ 2. |
| `MAX_TOKENS` | 8192 | Token budget for the whole response: thinking (generated plus inserted retrieval text) and the answer. |
| `THINK_FRACTION` | 0.75 | Share of `MAX_TOKENS` the thinking may use: 6144 of 8192, with the other 2048 reserved for the answer. **Why budgets instead of a turn cap:** how many searches to make is something the policy should learn. When the thinking reaches its budget, `</think>` is forced and the judge is told it was cut off. Over-long thinking therefore costs reward, but there is always room left to answer. Must be in [0.1, 0.95]. |
| `MAX_MODEL_LEN` | 9216 | vLLM context length (prompt plus response). Must be ≥ `MAX_TOKENS + 512`, leaving room for the prompt. |
| `TEMPERATURE` | 1.0 | Rollout sampling temperature. **Why 1.0:** the policy gradient assumes rollouts are samples from the policy whose log-probs the HF model computes, and that is the untempered distribution. Sampling at T ≠ 1 (or `TOP_P` < 1) draws from a different distribution and biases the update. T = 1 also supplies the exploration GRPO depends on. |
| `TOP_P` | 1.0 | Rollout nucleus sampling. Kept at 1.0 for the same reason. |
| `ROLLOUT_SEED_PER_STEP` | 1 | *config.py only.* Gives every rollout call a fixed seed derived from `SEED`, the step, the rollout and the turn. 0 leaves rollouts unseeded. |
| `EVAL_TEMPERATURE` | 0.6 | Held-out eval sampling temperature. **Why not greedy:** Qwen3's model card advises against greedy decoding in thinking mode because it tends to loop. 0.6 / 0.95 / 20 is its recommended setting. Seeds are fixed per question, so evals are comparable across steps. `0` gives greedy decoding. |
| `EVAL_TOP_P` | 0.95 | Held-out eval nucleus sampling. |
| `EVAL_TOP_K` | 20 | Held-out eval top-k. |

### vLLM (rollouts, judge, reference model)

| Variable | Default | What it does |
|---|---|---|
| `VLLM_GPU_MEM` | 0.80 | Fraction of the 33 GB slice vLLM takes while awake, for weights plus KV cache. **Why not higher:** the remaining ~6 GB holds what stays on the GPU while vLLM runs: the e5 encoder, CUDA contexts and allocator slack. Raising it buys KV cache but risks running out of memory. While vLLM sleeps it gives almost all of this back to the NF4 model. |
| `VLLM_QUANTIZATION` | `fp8` | Weight quantization of the vLLM copy. **Why:** fp8 roughly halves the ~16 GB of bf16 weights, which leaves enough KV cache to run a whole batch of long rollouts on one slice. **Cost:** vLLM's log-probs no longer match the NF4 training model (about 0.2 nats per token, the `gap` metric). That is why `REF_LOGPROBS_SOURCE` and `OLD_LOGPROBS_SOURCE` default to `hf`. |
| `VLLM_KV_CACHE_DTYPE` | `fp8` | KV cache precision. fp8 fits twice as many tokens as bf16. |
| `VLLM_MAX_NUM_SEQS` | 128 | Most sequences vLLM decodes at once. Kept ≥ the batch (64) and `EVAL_N` (100) so all of them run together. |
| `VLLM_ENFORCE_EAGER` | 0 | 1 skips CUDA-graph capture: faster startup, slower decoding. Useful when debugging. |
| `VLLM_SLEEP_LEVEL` | 1 | How vLLM sleeps during the gradient step. Level 1 parks the weights in CPU RAM, so waking is fast but uses RAM from the 72 GiB cgroup. Level 2 discards them and reloads from disk on wake: slower, but saves RAM. |

### LoRA and optimizer (the NF4 training copy)

| Variable | Default | What it does |
|---|---|---|
| `LORA_R` | 16 | LoRA rank. Also passed to vLLM as `max_lora_rank`. |
| `LORA_ALPHA` | 32 | LoRA scale. The adapter's output is multiplied by alpha / r (here 2). |
| `LORA_DROPOUT` | 0.0 | Dropout on the LoRA input. **Why 0:** dropout makes the training forward pass random. Log-probs, and with them the PPO ratio and the KL, would then differ from the rollout policy's before any update. |
| `LORA_TARGETS` | all attention and MLP projections | Linear layers that get adapters. **Why all of them:** the QLoRA paper (Dettmers et al., 2023) found that adapters on every linear layer are needed to match full fine-tuning. |
| `LR` | 2e-5 | AdamW learning rate on the LoRA weights. **Why so far above Search-R1's 1e-6:** that rate was for full fine-tuning. A LoRA update starts at zero and is low-rank, so it needs much larger steps to move the policy. |
| `LR_WARMUP_STEPS` | 10 | Linear warm-up, counted in **optimizer** steps. These equal GRPO steps only while `PPO_EPOCHS=1` and `MINIBATCH_SIZE=0`. |
| `LR_SCHEDULE` | `constant` | `constant`, or `cosine`, which decays to 0 at `TOTAL_STEPS × PPO_EPOCHS × minibatches per step`. |
| `WEIGHT_DECAY` | 0.0 | AdamW weight decay. |
| `MAX_GRAD_NORM` | 1.0 | Gradient-norm clipping threshold. `gnorm` logs the norm before clipping. |
| `HF_SWAP_MODE` | `offload` | How the NF4 model leaves the GPU between updates. `offload` moves it and the optimizer state to CPU RAM (about 4 s out, under 1 s back). `reload` frees it and rebuilds it from disk every step (slow, but always works). `keep` leaves it on the GPU, which is only viable with a small `VLLM_GPU_MEM`. At startup a round-trip test compares logits before and after offloading, and falls back to `reload` if offloading corrupted the 4-bit weights. |
| `HF_MICRO_BATCH_SIZE` | 2 | Sequences per forward/backward pass. **Why 2:** a 2 × 8000-token pass peaks at 18.1 GB. Gradients accumulate up to the minibatch, so this changes memory use, not the update. |
| `HF_LOGITS_CHUNK` | 1024 | Tokens per chunk for `lm_head` and the log-softmax. **Why:** with a 152k vocabulary, the logits for 2 × 6k tokens would take 3.6 GB in bf16 alone, more in fp32 and with gradients. Chunking inside an activation checkpoint keeps only one chunk alive at a time. |

### GRPO objective

| Variable | Default | What it does |
|---|---|---|
| `PPO_EPOCHS` | 1 | Gradient passes over each rollout batch. |
| `MINIBATCH_SIZE` | 0 | Sequences per optimizer step; 0 = the whole batch. Must be a multiple of `HF_MICRO_BATCH_SIZE`. **Why the defaults (1 and 0):** one optimizer step per batch keeps every update exactly on-policy. The trade-off is that `CLIP_EPS` never comes into play: the ratio is 1 at the only step, so `clip` stays 0. Raise either value to get more updates out of each expensive rollout batch; clipping then starts to matter. |
| `CLIP_EPS` | 0.2 | PPO clip range: the ratio π_new / π_old is clipped to [0.8, 1.2]. The standard PPO value. |
| `KL_COEF` | 0.001 | Weight of the KL penalty to the base model. **Why so small:** it is Search-R1's value. It stops the policy drifting far from the base model's language without fighting the reward. |
| `KL_TYPE` | `low_var_kl` | KL estimator. `low_var_kl` is the "k3" estimator, exp(d) − d − 1 with d = logp_ref − logp_new: always ≥ 0 and lower-variance than the plain log-ratio `kl`. It is the Search-R1/verl default. `mse` is ½ (logp_new − logp_ref)². |
| `ADV_NORM_STD` | 1 | Divide group-centred rewards by the group's standard deviation (standard GRPO). `0` gives Dr. GRPO (Liu et al., 2025), which drops the division because it up-weights groups whose rewards barely vary, i.e. questions that are almost all right or almost all wrong. |
| `LOSS_AGG` | `token-mean` | How per-token losses are averaged. `token-mean` weights every generated token in the minibatch equally (DAPO-style), so long rollouts are not diluted. `seq-mean-token-mean` averages within each sequence, then across sequences (the original GRPO), which gives each token of a long rollout less weight than a token of a short one. |
| `REF_LOGPROBS_SOURCE` | `hf` | Source of the reference log-probs for the KL term. **Why `hf`:** the reference should be the training model minus the LoRA, so that KL measures only what the LoRA changed. `hf` runs the NF4 model with adapters disabled: same numerics as the policy, so KL is exactly 0 at step 1. `vllm` gets them for free from the fp8 model via `prompt_logprobs`, but fp8 and NF4 quantize differently. That difference alone reads as KL ≈ 0.2 per token, and the KL term would pull the policy toward fp8's rounding noise instead of toward the base model. **Cost of `hf`:** one extra no-grad forward pass per update. |
| `OLD_LOGPROBS_SOURCE` | `hf` | Source of the "old" log-probs in the PPO ratio exp(logp_new − logp_old). **Why `hf`:** "new" always comes from the NF4 model, so "old" must come from the same model, or the ratio is wrong before any update has happened. With `vllm` (the fp8 rollout log-probs, free) the ~0.2 nats/token gap puts ratios around e^±0.2 ≈ 0.82–1.22, right at the 0.2 clip boundary. Many tokens would then be clipped for reasons unrelated to learning. `hf` recomputes them in one no-grad pass, so the ratio starts at exactly 1, and it also measures the HF-vs-vLLM difference, which is logged as `gap`. |

### Reward

`reward = Q − penalty`, where `c` is the average probability the policy gave its own generated tokens and
`ramp(x, lo, hi)` rises linearly from 0 at `lo` to 1 at `hi`. See [src/grpo/reward.py](src/grpo/reward.py).

| Variable | Default | What it does |
|---|---|---|
| `W_UNDER` | 0.5 | Largest penalty for answering **without searching** while unsure: `W_UNDER × (1 − ramp(c, UNDER_LO, UNDER_HI))`. **Why the larger weight:** answering from memory with low confidence is exactly when the agent should have searched, and this is the main behaviour the reward shapes. |
| `UNDER_LO` / `UNDER_HI` | 0.3 / 0.7 | Below c = 0.3 the full under-confidence penalty applies, above 0.7 none, linear in between. Skipping the search costs nothing only if the model was confident. |
| `W_OVER` | 0.1 | Largest penalty for **over-confidence after searching** on a good answer: `W_OVER × ramp(c, OVER_LO, OVER_HI)^PEN_POWER`. **Why small:** it is a calibration nudge, not the main signal. It can reorder rollouts within a group without overriding Q. |
| `OVER_LO` / `OVER_HI` | 0.5 / 1.0 | Confidence range over which the over-confidence penalty ramps up. |
| `PEN_POWER` | 2 | Exponent on that ramp. **Why 2:** squaring keeps the penalty tiny at moderate confidence and puts it near saturation, which is what it is meant to discourage: c = 0.75 costs 0.025, c = 0.95 costs 0.081. |
| `Q_GATE` | 0.7 | The over-confidence penalty applies only when Q ≥ 0.7. **Why:** poor answers already score low through Q, so the gate keeps the calibration nudge on answers that are otherwise good. |
| `FORMAT_PENALTY` | 0.0 | Extra penalty when no `<answer>` is produced. Off by default because the judge's structure score already penalizes a missing answer. |

### Judge

| Variable | Default | What it does |
|---|---|---|
| `JUDGE_USE_GOLD` | 0 | Whether the judge sees the HotpotQA gold answer. **Off (default):** Q grades structure, reasoning and the form of the answer only. Correctness is never rewarded directly, and held-out EM/F1 show whether that is enough. **On:** a wrong answer is capped at Q ≤ 0.3, so correctness dominates the reward. |
| `JUDGE_MAX_TOKENS` | 16 | Generation cap for the judge, which replies with a single number. Replies with no parsable number are scored 0.5 and counted in `judge_parse_failures`. |
| `JUDGE_MAX_CHARS` | 12000 | *config.py only.* The transcript shown to the judge is clipped to this many characters, keeping the first third and the last two thirds (where the answer is). Each `<information>` block is first cut to 1500 characters. |

### Schedule and checkpoints

| Variable | Default | What it does |
|---|---|---|
| `TOTAL_STEPS` | 300 | Number of GRPO steps. |
| `EVAL_EVERY` | 20 | Held-out eval every N steps, plus once at step 0 (the base model). 0 disables it. |
| `EVAL_N` | 100 | Held-out dev questions, one rollout each. Each question is worth 1 / `EVAL_N` of EM, so at 100 EM moves in steps of 0.01. |
| `SAVE_EVERY` | 20 | Steps between optimizer-state saves (`adapters/step_N/optimizer.pt`); the last step always saves. Adapter weights are saved every step and only the newest 2 are kept, but adapters that hold optimizer state are never pruned. Those are the resume points. |
| `KEEP_LAST` | 3 | Number of `checkpoints/step_N/` directories (one per eval) to keep. `best/` is never pruned. |
| `BEST_METRIC` | `em` | Eval metric that decides `checkpoints/best/`: `em`, `f1` or `reward`. |
| `SAMPLE_DUMP_EVERY` | 5 | Every N steps, write the best, the worst and 2 random trajectories to `logs/samples/step_N.jsonl`. |
| `SEED` | 0 | Seeds Python, NumPy and torch, the question order, the held-out draw, the rollout and eval sampling seeds, and vLLM. |
| `LOG_LEVEL` | `INFO` | Python log level for `logs/train_*.log` and the console. |

### Process environment

| Variable | Default | What it does |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | 0 | The GPU (the MIG slice). |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | 16 | CPU threads for torch and NumPy. |
| `TOKENIZERS_PARALLELISM` | `false` | Avoids the HF tokenizers' fork warning and deadlock once vLLM starts other processes. |
| `HF_HUB_OFFLINE` | 1 | All weights are local, so the Hugging Face Hub is never contacted. |
| `VLLM_LOGGING_LEVEL` | `WARNING` | Keeps vLLM's per-request logging out of the console. |
| `VLLM_USE_FLASHINFER_SAMPLER` | 0 | FlashInfer's sampler is JIT-compiled with nvcc, which the pod doesn't have, so vLLM uses torch sampling instead. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Less memory fragmentation with variable-length batches, while vLLM and the HF model take turns on the same 33 GB. |
| `PYTHONUNBUFFERED` | 1 | Unbuffered output, so the tee'd console log is written as it happens. |

## Metrics reference

| Where | What |
|---|---|
| console | One row per step, an `EVAL` line per evaluation and `WARN` lines. Copied to `logs/train_*.log`; `train_rl.sh` also tees the whole console into `logs/console_*.log`. |
| `logs/metrics.jsonl` | One record per step: everything in the row and more. |
| `logs/eval.jsonl` | One record per evaluation. Also saved as `checkpoints/step_N/metrics.json`. |
| `logs/samples/step_N.jsonl` | Full best, worst and random trajectories every `SAMPLE_DUMP_EVERY` steps. |
| `checkpoints/step_N/samples.jsonl` | Every trajectory of that evaluation. |

The console shows rates as percentages; the JSON files store them as fractions in [0, 1]. A `-` in the row means the value
wasn't computed that step, e.g. because the update was skipped.

### Console row (one per step)

| Column | Meaning |
|---|---|
| `step` | GRPO step number, starting at 1. The eval at step 0 is the model before any update. |
| `reward` | Mean reward over the batch: `Q − pen`, minus `FORMAT_PENALTY` when there's no answer. |
| `±` | Standard deviation of the reward across the whole batch. This is not the within-group spread GRPO learns from; that is `group_reward_std_mean` in `metrics.jsonl`. |
| `Q` | Mean judge score in [0, 1]. The base model (adapter off, thinking off) grades structure (up to 0.35), reasoning (up to 0.35) and the answer (up to 0.30). It does not check correctness unless `JUDGE_USE_GOLD=1`. |
| `c` | Mean confidence: the average probability the policy gave to the tokens it generated, from the vLLM rollout log-probs, with inserted `<information>` text excluded. It drives the penalty. |
| `pen` | Mean confidence penalty (see [Reward](#reward)). |
| `srch%` | Percentage of rollouts with at least one search. |
| `n_srch` | Mean searches per rollout, counting searches whose results were inserted. |
| `ans%` | Percentage of rollouts that answered after `</think>`, the correct protocol. An answer given inside the thinking block doesn't count (see `answer_in_think_rate`). |
| `cut%` | Percentage of rollouts whose thinking reached `THINK_FRACTION × MAX_TOKENS` and was force-closed. If this stays high, the policy isn't learning to be brief or the budget is too small. |
| `gen` | Mean tokens the model generated per rollout. Inserted `<information>` text and the forced `</think>` are excluded. |
| `kl` | Mean per-token KL estimate (`KL_TYPE`) between the policy and the reference. 0 at step 1 because the LoRA starts at zero; it should then grow slowly. |
| `pg_loss` | Policy-gradient part of the loss, averaged per token. **Not "lower is better".** On the first pass it equals minus the token-weighted mean advantage, so its sign mostly says whether the higher-advantage rollouts were longer or shorter. |
| `clip` | Fraction of tokens whose ratio fell outside [1 − `CLIP_EPS`, 1 + `CLIP_EPS`]. Always about 0 with `PPO_EPOCHS=1` and `MINIBATCH_SIZE=0` (see [GRPO objective](#grpo-objective)). |
| `ent` | Mean per-token entropy of the policy over the generated tokens, in nats, from the training pass. A slide toward 0 means the policy is collapsing to deterministic output. |
| `gnorm` | Gradient norm before clipping to `MAX_GRAD_NORM`, averaged over the step's optimizer steps. |
| `zeroG` | Fraction of groups whose rollouts all got the same reward. Their advantages are 0, so they contribute nothing to the update. |
| `gap` | Mean absolute difference, per generated token, between the log-probs of the NF4 training model and the fp8 rollout model on the same tokens. About 0.2 is normal quantization noise. Much higher usually means vLLM isn't running the current adapter. Only computed with `OLD_LOGPROBS_SOURCE=hf`. |
| `t_roll` | Seconds of rollout generation and retrieval. |
| `t_train` | Seconds of the update (including the reference and old log-prob passes) plus saving the adapter. |
| `t_step` | Seconds for the whole step: also the judge, vLLM reference log-probs if used, and both GPU swaps. Excludes eval. |

### `EVAL` line

Each held-out question gets one rollout, sampled with the `EVAL_*` settings and a fixed seed per question.

| Field | Meaning |
|---|---|
| `step` | Step whose adapter was evaluated; 0 is the base model. |
| `n` | Number of held-out questions (`EVAL_N`). |
| `EM` | Exact match with the gold answer after HotpotQA normalisation (lowercase, punctuation and articles removed, whitespace collapsed). The default `BEST_METRIC`. |
| `F1` | Token-overlap F1 with the gold answer. It gives partial credit. |
| `Q`, `reward` | As in training, computed on the eval rollouts. |
| `ans%` | Percentage of questions with **any** `<answer>`, including one given inside the thinking block. This is looser than the training `ans%`. |
| `cut%`, `srch%`, `n_srch` | As in training. |
| `any-gold` | Fraction of questions where at least one gold supporting article's title appears among everything retrieved in the trajectory. It counts all `RETRIEVE_TOPK` passages per search, including those beyond `CONTEXT_TOPK` that the model never saw. |
| `all-gold` | Fraction where all the gold articles appear (usually 2 in HotpotQA), i.e. both hops were found. Same counting as `any-gold`. |
| `Ns` | Wall-clock time of the eval in seconds. |
| `** new best **` | `BEST_METRIC` strictly beat the best so far, and the checkpoint was copied to `checkpoints/best/`. Step 0 is always a new best. |

### `WARN` lines

After every step, [dashboard.py](src/train_harness/dashboard.py) checks the last 20 rows.

| Warning | Fires when | Usually means |
|---|---|---|
| no learning signal | `±` < 0.02 for 5 steps | Every reward has collapsed to the same value. |
| identical groups | mean `zeroG` > 0.75 over 5 steps | Most groups teach nothing. Raise `ROLLOUTS_PER_GROUP` or check the judge. |
| stopped searching | `srch%` < 5 % for 5 steps, after step 10 | The policy learned to answer from memory. Check `W_UNDER`. |
| always searching | `srch%` > 98 % for 5 steps, after step 10 | Searching is never penalized. Check `W_OVER` / `Q_GATE`. |
| few answers | `ans%` < 30 % for 5 steps | The answer budget is too small, or the output format has collapsed. |
| thinking cut off | `cut%` > 90 % for 5 steps | The policy isn't learning to be brief, or `MAX_TOKENS` is too small. |
| near-deterministic | `c` > 0.97 for 3 steps | Entropy collapse. |
| entropy collapse | `ent` < 0.1 for 3 steps | Entropy collapse. |
| KL spike | `kl` > max(1, 10 × its running median) | An unstable update. |
| large gradient | `gnorm` > 50 or not finite | An unstable update. |
| train/inference mismatch | `gap` > 0.5 | vLLM is probably not running the new adapter. |
| reward plateau | mean reward of the last 20 steps within 0.005 of the 20 before, after step 60 | Learning has stalled. |

### `logs/metrics.jsonl` (one record per step)

**Rollouts** ([harness.py](src/train_harness/harness.py))

| Key | Meaning |
|---|---|
| `ts` | Unix time the record was written. |
| `step`, `n_rollouts` | Step number and batch size. |
| `reward_mean`, `reward_std`, `reward_min`, `reward_max` | Reward over the batch. The first two are `reward` and `±`. |
| `quality_mean` | `Q`. |
| `confidence_mean`, `confidence_std` | `c`, and its spread across rollouts. |
| `penalty_mean` | `pen`. |
| `search_rate`, `n_searches_mean` | `srch%`, `n_srch`. |
| `answer_rate` | `ans%` (answered after `</think>`). |
| `answer_in_think_rate` | Fraction that wrote `<answer>` inside the thinking block. This breaks the protocol, and the judge is told. |
| `think_truncated_rate` | `cut%`. |
| `think_closed` | Counts of how the thinking ended: `model` (closed by the model), `budget` (cut off), `eos` (the model stopped without closing it, so it was force-closed), `never` (answered inside it). |
| `finish` | Counts of how rollouts ended: `answer`, `answer_in_think`, `no_answer`. |
| `think_tokens_mean` | Mean thinking-block length, generated plus inserted tokens. |
| `answer_tokens_mean` | Mean answer length. |
| `n_turns_mean` | Mean generation calls per rollout: one per thinking segment (searches + 1) plus one for the answer. |
| `gen_tokens_mean` | `gen`. |
| `resp_tokens_mean`, `resp_tokens_max` | Response length including inserted tokens, which is what the update has to process. |
| `n_retrievals` | Queries sent to the retriever this step. |
| `retrieval_ms_last` | Latency of the step's last batched retriever call. |
| `judge_parse_failures` | Judge replies with no number, scored 0.5. **Cumulative** since the process started, eval included. |
| `sampler` | Position in the training set: `epoch`, `cursor`, `pool_size`. |

**Advantages** ([advantage.py](src/grpo/advantage.py))

| Key | Meaning |
|---|---|
| `adv_mean` | About 0 by construction. |
| `adv_std`, `adv_abs_mean` | Size of the learning signal. Both shrink as more groups have identical rewards. |
| `group_reward_std_mean` | Mean within-group reward standard deviation: the reward differences GRPO actually learns from. `±` also includes differences between questions. |
| `frac_zero_var_groups` | `zeroG`. |
| `n_groups` | Number of groups in the batch. |

**Update** ([backprop.py](src/grpo/backprop.py), [loss.py](src/grpo/loss.py)). These are token-weighted means over all micro-batches.
If every advantage is 0, no update is made: the record has `skipped: true` and none of the other keys in this table.

| Key | Meaning |
|---|---|
| `skipped` | True when the update was skipped. |
| `pg_loss`, `kl`, `clipfrac`, `entropy`, `grad_norm`, `lp_gap_hf_vllm` | The row's `pg_loss`, `kl`, `clip`, `ent`, `gnorm`, `gap`. |
| `loss` | Total loss per token: `pg_loss + KL_COEF × kl`. |
| `ratio_mean` | Mean of π_new / π_old. 1 on the first pass with `OLD_LOGPROBS_SOURCE=hf`. |
| `approx_kl_old` | ½ (logp_new − logp_old)² per token: how far the policy moved during this update. 0 when there is one optimizer step per batch. |
| `grad_norm_max` | Largest pre-clip gradient norm among the step's optimizer steps. |
| `lr` | Learning rate after the last optimizer step. |
| `optimizer_steps` | Optimizer steps taken: `PPO_EPOCHS × ceil(batch / MINIBATCH_SIZE)`, minus any skipped for a non-finite gradient. |
| `n_tokens_trained` | Generated tokens that went into the loss, summed over epochs. |
| `prep_s` | Seconds of the no-grad reference and old log-prob passes. |
| `train_s` | Seconds of the whole update, preparation included. |

**Timing and system** ([harness.py](src/train_harness/harness.py))

| Key | Meaning |
|---|---|
| `t_rollout`, `t_train`, `t_step` | The row's `t_roll`, `t_train`, `t_step`. |
| `t_judge` | Judge scoring. |
| `t_ref` | Reference log-probs from vLLM. About 0 with `REF_LOGPROBS_SOURCE=hf`, where that pass is part of `prep_s` instead. |
| `t_swap_to_hf` | vLLM going to sleep and the NF4 model moving onto the GPU. |
| `t_swap_to_vllm` | The NF4 model moving off the GPU, vLLM waking up and loading the new adapter. |
| `rollout_generate_s`, `rollout_retrieval_s` | `t_rollout` split between vLLM generation and retrieval. |
| `vllm_sleep_s`, `vllm_wake_s`, `vllm_judge_s`, `vllm_ref_s` | vLLM time by call type since the previous record. The step right after an eval also includes that eval's judge calls. |
| `free_gpu_gb` | Free GPU memory at the end of the step, with vLLM awake. A steady decline points to a leak. |

### `logs/eval.jsonl` and `checkpoints/step_N/metrics.json`

| Key | Meaning |
|---|---|
| `step`, `n`, `em`, `f1`, `quality_mean`, `reward_mean`, `answer_rate`, `think_truncated_rate`, `search_rate`, `n_searches_mean`, `retrieval_any_gold`, `retrieval_all_gold`, `eval_s` | The `EVAL` line's fields, in order. |
| `confidence_mean`, `penalty_mean` | As in training. |
| `gen_tokens_mean`, `think_tokens_mean` | Mean generated tokens and mean thinking-block length. |
| `answer_in_think_rate`, `finish` | As in training. |
| `rollout_generate_s`, `rollout_retrieval_s` | The eval's time split between generation and retrieval. |
| `adapter` | Path of the adapter that was evaluated. |
| `is_best` | Whether this eval set a new best `BEST_METRIC`. |

### Trajectory dumps

Each line of `logs/samples/step_N.jsonl` and `checkpoints/step_N/samples.jsonl` is one trajectory.

| Key | Meaning |
|---|---|
| `tag`, `step` | `best`, `worst` or `random`, and the step number. Only in `logs/samples/`. |
| `qid`, `question`, `gold` | HotpotQA id, the question and the gold answer. |
| `group` | Index of the trajectory's group within the step. |
| `answer` | The extracted final answer, or null if there was none. |
| `finish`, `think_closed` | How the rollout and its thinking ended (see `metrics.jsonl`). |
| `n_turns`, `n_searches`, `searches` | Generation calls, searches whose results were inserted, and every query issued. |
| `retrieved_titles` | Titles of the passages returned for each search, all `RETRIEVE_TOPK` of them. |
| `think_tokens`, `answer_tokens`, `n_resp`, `n_gen` | Thinking length, answer length, and response length with and without inserted tokens. |
| `confidence`, `quality`, `penalty`, `reward`, `advantage` | The rollout's c, Q, penalty, reward and group-relative advantage. `advantage` is NaN in eval dumps. |
| `text` | The full decoded response, special tokens included. |

```
dataset/
  load_index.py       open the IVF-OPQ index (mmap), the flat e5 file (pread) and the passage parquet
  move_pages.py       prefetch inverted lists, fetch exact vectors / passage text, page-cache accounting
src/
  config.py           TrainConfig: every knob, each settable through an environment variable
  main.py             entry point (python -m src.main), flags: --print-config --check-retrieval --smoke
  retriever/dense_retriever.py   e5 query encoder; coarse -> PQ scan -> exact re-rank -> passages
  engine/prompts.py   Search-R1 instruction for native thinking, <information> formatting, tag parsing
  engine/vllm.py      VLLMEngine: rollouts (LoRA), reference log-probs, judge chat, sleep/wake
  engine/rollout.py   think phase (generate -> <search> -> retrieve -> splice -> ...) under a token budget, then answer
  engine/local_llm.py LocalPolicy: NF4 Qwen3-8B + LoRA, log-probs with grad, CPU<->GPU swap, save adapter
  grpo/reward.py      reward = judge Q - confidence penalty (toy train_grpo.py definition)
  grpo/advantage.py   group-normalised advantages
  grpo/judge.py       LLM judge prompt and parsing (same vLLM model, adapter off)
  grpo/loss.py        clipped PG surrogate + KL (low_var_kl / kl / mse)
  grpo/backprop.py    GRPOUpdater: epochs x minibatches x micro-batches, old/ref log-probs, clipping
  train_harness/
    harness.py        the loop (sampling, rollouts, judge, reward, swap, update, swap, log, eval)
    sampling.py       HotpotQA pools, uniform random group assignment, held-out split
    checkpoints.py    held-out eval (EM/F1/retrieval coverage/judge), best/ + step_N checkpoints
    dashboard.py      one console row per step + health warnings (stall / collapse heuristics)
    logging.py        text log, metrics.jsonl, eval.jsonl, trajectory sample dumps
scripts/
  setup_env.sh        conda env + pinned installs (see "Environment")
  train_rl.sh         all environment variables with defaults; runs python -m src.main
output/rl/            logs/, adapters/, checkpoints/ (git-ignored)
```

## Run

```bash
bash scripts/setup_env.sh                 # once
bash scripts/train_rl.sh --check-retrieval "Were Scott Derrickson and Ed Wood of the same nationality?"
bash scripts/train_rl.sh --smoke          # 2 groups x 2 rollouts x 2 steps, writes output/rl_smoke/
bash scripts/train_rl.sh                  # the real thing; override anything: NUM_GROUPS=16 LR=1e-5 bash scripts/train_rl.sh
RESUME_FROM=output/rl/adapters/step_00100 bash scripts/train_rl.sh
```

Data lives in `~/datasets` (override with `DATASETS_DIR`): `wiki-18/compressed/*.index|*.parquet`,
`wiki-18/raw/wiki-18-e5-index/e5_Flat.index`, `hotpotqa/hf/distractor/*.parquet`, `models/Qwen3-8B`,
`models/e5-base-v2`.

## One GRPO step

1. `NUM_GROUPS` HotpotQA questions are drawn uniformly without replacement; each starts `ROLLOUTS_PER_GROUP` rollouts.
2. Rollouts run on vLLM with the current adapter. The prompt ends with an open `<think>` block (Qwen3's native
   thinking). Generation stops at `</search>`, `</think>` or `</answer>`. A search is embedded with e5, the IVF-OPQ
   index returns the top-`RERANK_CANDIDATES` codes from `NPROBE` lists, their exact vectors are read from the flat
   file and re-scored, and the top-`CONTEXT_TOPK` passages are spliced into the think block as
   `<information>Doc i(Title: ...) ...</information>`; the model keeps thinking. There is no turn cap: the think
   block (generated + inserted tokens) may use `THINK_FRACTION` x `MAX_TOKENS`, then `</think>` is forced and the
   rest of `MAX_TOKENS` is for the answer. One `<information>` block is capped at `MAX_OBS_TOKENS`. Inserted tokens
   carry `response_mask = 0` and never enter the loss.
3. The same vLLM model with the adapter off scores every trajectory as a judge (Q in [0, 1]). Reference log-probs
   for the KL term come from the NF4 model with adapters disabled (`REF_LOGPROBS_SOURCE=hf`, one extra no-grad
   pass in step 6); `=vllm` takes them from the fp8 model via `prompt_logprobs` instead, which is free but the
   fp8-vs-NF4 difference alone (~0.2 nats/token) then shows up as KL even when the LoRA is zero.
4. `reward = Q - penalty(c, searched, Q)` where `c` is the mean probability of the generated tokens (from the rollout
   logprobs); advantages are group-centred and (by default) divided by the group std.
5. vLLM sleeps (fp8 weights to CPU RAM, KV cache freed); the NF4 model comes back to the GPU.
6. `PPO_EPOCHS` passes over the batch in `MINIBATCH_SIZE` chunks: fresh log-probs with grad, clipped ratio against the
   pre-update log-probs (`OLD_LOGPROBS_SOURCE=hf` recomputes them with this model so the ratio is exactly 1 on the first
   pass; the mean |hf - vLLM| gap is logged as `gap`), `KL_COEF` x low-variance KL, grad-norm clipping, AdamW on the LoRA.
7. Adapter saved to `output/rl/adapters/step_N`; NF4 model back to CPU; vLLM wakes and loads the new adapter.
8. Metrics row printed and appended to `logs/metrics.jsonl`; every `EVAL_EVERY` steps the policy answers
   `EVAL_N` held-out dev questions (sampled with the `EVAL_*` settings and fixed seeds) (EM, F1, answer/search rates, gold-article coverage of what was retrieved, judge Q)
   and the adapter is copied to `checkpoints/step_N` and, if `BEST_METRIC` improved, `checkpoints/best`.

## Choices worth knowing

* **Prompt**: Search-R1's instruction (single user message, no SFT) reworded for one native thinking block: search
  inside `<think>`, answer after `</think>`. Search-R1's models had no thinking mode and wrote `<think>` as text;
  Qwen3 uses the same tokens for its native block, so that block is the reasoning channel here.
* **Budgets, not turns**: `MAX_TOKENS=8192`, `THINK_FRACTION=0.75` -> 6144 tokens of thinking (searches included),
  2048 for the answer; `MAX_MODEL_LEN=9216` leaves room for the prompt. How many searches fit is for the policy to learn; a cut-off is force-closed and reported to
  the judge, so over-long thinking costs reward. Held-out eval samples with Qwen3's recommended T=0.6 / top-p 0.95 /
  top-k 20 and fixed seeds (`EVAL_*`), since greedy decoding in thinking mode tends to loop.
* **Context size**: the retriever returns 10 exact-reranked passages; 5 go into `<information>` (Search-R1 uses 3;
  more helps the second hop of HotpotQA, 10 would cost ~1.3k tokens per search). `CONTEXT_TOPK` changes it.
* **Judge** is the fp8 base model in vLLM (no extra GPU memory), with thinking disabled and a deterministic format
  summary (think blocks, searches, answer tag, invalid actions, truncation) in its prompt to anchor the structure score.
  The **KL reference** defaults to the NF4 model with adapters disabled (see step 3 above).
* **Memory**: vLLM owns `VLLM_GPU_MEM` (0.80) of the slice while awake; asleep it frees everything but its CUDA
  context. The NF4 model is 6.3 GB resident; a 2 x 8000-token forward/backward peaks at 18.1 GB because the
  decoder runs under activation checkpointing and `lm_head` + log-softmax run per `HF_LOGITS_CHUNK`-token chunk
  inside a checkpoint, so the (B, T, 152k) logits never exist as one tensor. Offload to CPU: ~4 s, restore: <1 s.
  Note: `peft.prepare_model_for_kbit_training` is deliberately not used (it would upcast `lm_head` to fp32).
* **Retrieval accuracy** (HotpotQA-dev, question-only): nprobe 128 + exact re-rank of 100 gives recall@10 0.81 vs
  the 64 GB flat index, any-gold@10 0.77 (exact: 0.82), ~20 ms per query warm.

## Environment

The pod's driver is 570 (CUDA 12.8). vLLM >= 0.28 pins torch 2.13 (CUDA 13 wheels, need an R580 driver), so
`scripts/setup_env.sh` installs vLLM 0.26.0's `+cu129` wheel with torch 2.11.0+cu129 (CUDA minor-version
compatibility), peft 0.21.2, bitsandbytes 0.50.2 and faiss-cpu 1.13.2 (1.15.1 corrupts batched IVF search)
into the conda env `grpo-rl`, and removes the CUDA-13 `torchcodec` that breaks `import vllm`. There is no CUDA
toolkit on the pod, so `VLLM_USE_FLASHINFER_SAMPLER=0` keeps vLLM off the one kernel it would JIT-compile with nvcc.
