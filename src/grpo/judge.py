"""LLM-as-judge quality score Q in [0, 1] for a whole trajectory.

The judge is the same fp8 Qwen3-8B that vLLM already serves, without the LoRA adapter and
with its own thinking disabled, so it costs one short chat completion per rollout and no extra
GPU memory. It grades the *process* (one self-closed think block, non-degenerate reasoning,
distinct and purposeful searches, grounding in the retrieved passages) and the *answer*
(concise, direct, after the thinking). A computed format check (thinking cut off at the budget?
answer inside the block? repeated queries?) is included so structure is not left to the judge's
reading of a long transcript. With JUDGE_USE_GOLD=1 the HotpotQA gold answer is shown and
factual accuracy dominates the score.
"""
from __future__ import annotations

import logging
import re

import numpy as np

from src.config import TrainConfig
from src.engine.rollout import Trajectory, think_budget
from src.engine.vllm import VLLMEngine

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a strict, criteria-based judge for a retrieval-augmented question-answering agent. "
    "The agent thinks inside one <think>...</think> block. While thinking it may search with <search>query</search>, "
    "after which the top results appear inside <information>...</information> and it keeps reasoning. After </think> it "
    "must give one short final answer inside <answer>...</answer>. "
    "You will be shown the question, the agent's full transcript and a computed format check. Rate the transcript from 0.0 to 1.0.\n\n"
    "Scoring rubric (sum of the parts, then clamp to [0.0, 1.0]):\n"
    "1. Structure (0.0-0.35): the agent closes its own thinking (it was not cut off at the token budget and did not stop "
    "without closing it); search tags are well formed; there is exactly one final <answer> after the thinking, short, with "
    "nothing else around it; no answer attempts inside the thinking, no dangling tags.\n"
    "2. Reasoning quality and diversity (0.0-0.35): the reasoning progresses step by step without repeating itself or "
    "looping; it is concise rather than padded; search queries are distinct, specific and purposeful (no near-duplicates); "
    "retrieved information is actually read and used, not ignored.\n"
    "3. Answer quality (0.0-0.30): the final answer directly addresses the question, is as short as the question allows "
    "(an entity, a date, yes/no, ...), and is supported by the retrieved information when there is any.\n"
)

GOLD_ADDENDUM = (
    "4. Factual accuracy: a reference answer is provided. If the final answer does not match the reference in meaning, "
    "the total score must not exceed 0.3 regardless of structure and reasoning. If it matches, score the other criteria normally.\n"
)

OUTPUT_RULE = "\nRespond ONLY with a single floating-point number between 0.0 and 1.0. No words, no explanation."

_FLOAT_RE = re.compile(r"([0-9]*\.[0-9]+|[0-9]+)")
_INFO_RE = re.compile(r"(<information>)(.*?)(</information>)", re.DOTALL)
_SPECIAL_RE = re.compile(r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>")


class Judge:
    def __init__(self, engine: VLLMEngine, cfg: TrainConfig):
        self.engine = engine
        self.cfg = cfg
        self.n_parse_failures = 0

    def build_messages(self, traj: Trajectory) -> list[dict]:
        sys = SYSTEM_PROMPT + (GOLD_ADDENDUM if self.cfg.judge_use_gold else "") + OUTPUT_RULE
        text = compress_transcript(traj.text, self.cfg.judge_max_chars)
        user = f"Question: {traj.question.strip()}\n"
        if self.cfg.judge_use_gold:
            user += f"Reference answer: {traj.gold_answer}\n"
        user += f"\nAgent transcript:\n{text}\n\n{format_facts(traj, self.cfg)}\n\nScore:"
        return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    def score(self, trajs: list[Trajectory]) -> np.ndarray:
        if not trajs:
            return np.zeros(0)
        convs = [self.build_messages(t) for t in trajs]
        texts = self.engine.chat(convs, max_tokens=self.cfg.judge_max_tokens, temperature=0.0, use_lora=False)
        out = np.empty(len(trajs), dtype=np.float64)
        fails = 0
        for i, s in enumerate(texts):
            v = parse_score(s)
            if v is None:
                fails += 1
                v = 0.5
            out[i] = v
        if fails:
            self.n_parse_failures += fails
            log.warning("judge: %d/%d outputs had no parsable score (defaulted to 0.5); e.g. %r", fails, len(texts),
                        next((s for s in texts if parse_score(s) is None), ""))
        return out


def format_facts(traj: Trajectory, cfg: TrainConfig) -> str:
    """Deterministic protocol facts, so the structure part of the rubric does not rest on an 8B judge
    noticing a cut-off or a misplaced tag in a long transcript."""
    budget = think_budget(cfg)
    closed = {"model": "closed by the agent itself", "budget": "CUT OFF at the thinking token budget (forced closed)",
              "eos": "the agent stopped generating without closing it (forced closed)",
              "": "never closed: the agent answered inside the thinking block"}[traj.think_closed]
    distinct = len({" ".join(s.lower().split()) for s in traj.searches})
    if traj.finish == "answer":
        ans = f"yes, after the thinking -> {traj.answer!r}"
    elif traj.finish == "answer_in_think":
        ans = f"given INSIDE the thinking block (protocol violation) -> {traj.answer!r}"
    else:
        ans = "NO final <answer>"
    lines = [
        "Format check (computed, authoritative):",
        f"- thinking: {traj.think_tokens} tokens of a {budget}-token budget; {closed}",
        f"- searches: {traj.n_searches}" + (f" ({distinct} distinct)" if traj.n_searches else "")
        + (" -> repeated queries" if traj.n_searches and distinct < traj.n_searches else ""),
        f"- final answer: {ans}",
        f"- answer length: {traj.answer_tokens} tokens" if traj.answer_tokens else "- answer phase: not reached",
    ]
    return "\n".join(lines)


def parse_score(text: str) -> float | None:
    m = _FLOAT_RE.search(text.strip())
    if not m:
        return None
    v = float(m.group(1))
    if v > 1.0 and v <= 10.0:       # "8" or "7.5" on a ten-point habit
        v /= 10.0
    return min(1.0, max(0.0, v))


def compress_transcript(text: str, max_chars: int, info_chars: int = 1500) -> str:
    """Drop chat special tokens, shorten long <information> blocks, then clip the whole thing while
    keeping its end (where the answer is)."""
    text = _SPECIAL_RE.sub("", text).strip()
    text = _INFO_RE.sub(lambda m: m.group(1) + (m.group(2) if len(m.group(2)) <= info_chars
                                                 else m.group(2)[:info_chars] + " ...[truncated]") + m.group(3), text)
    if len(text) > max_chars:
        head, tail = max_chars // 3, max_chars - max_chars // 3
        text = text[:head] + "\n...[transcript truncated]...\n" + text[-tail:]
    return text
