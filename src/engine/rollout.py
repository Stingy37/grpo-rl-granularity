"""Rollouts with Qwen3's native thinking block as the reasoning channel (see prompts.py).

    prompt<think>\\n | reasoning <search> q </search> | \\n\\n<information>Doc 1(..)..</information>\\n\\n | reasoning </think> | \\n\\n<answer> a </answer>
                    |------- generated (mask 1) -----|------------- inserted (mask 0) -------------------|-- generated (1) --|---- generated (1) ----|
                    <------------------------- thinking: THINK_FRACTION x MAX_TOKENS ---------------------------------------> <-- answer: the rest -->

There is no cap on the number of searches or turns; the policy learns that. The only limits are
token budgets: the think block (generated + inserted tokens) may use THINK_FRACTION * MAX_TOKENS;
when that is exhausted "\\n</think>\\n\\n" is inserted (mask 0) and the model must answer with the
remaining (1 - THINK_FRACTION) * MAX_TOKENS. One <information> block is capped at MAX_OBS_TOKENS so
a single retrieval cannot eat the whole thinking budget. Generation stops at </search> (retrieve
and continue), </think> (go answer) or </answer>; an EOS inside the block also forces it closed.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from src.config import TrainConfig
from src.engine.prompts import (ANSWER_STOPS, THINK_CLOSE_FORCED, THINK_STOPS, encode_prompt, extract_answer,
                                format_information, last_search_query)
from src.engine.vllm import VLLMEngine
from src.retriever.dense_retriever import DenseRetriever, Passage, format_passages

log = logging.getLogger(__name__)

MIN_GEN_TOKENS = 16      # a generation call with less room than this is not made
MIN_INFO_TOKENS = 64     # a retrieval whose block could not get at least this many tokens is skipped
LEN_RESERVE = 2          # keep prompt+response < max_model_len so prompt_logprobs (max_tokens=1) always fits


@dataclass
class Trajectory:
    qid: str
    question: str
    gold_answer: str
    gold_titles: list[str]
    group: int
    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)       # 1 generated, 0 inserted
    gen_logprobs: list[float] = field(default_factory=list)      # rollout-time logprob, 0.0 where inserted
    text: str = ""
    phase: str = "think"                                         # think | answer | done
    n_turns: int = 0                                             # generation calls
    n_searches: int = 0
    searches: list[str] = field(default_factory=list)
    retrieved: list[list[Passage]] = field(default_factory=list)
    think_tokens: int = 0                                        # tokens in the think block (generated + inserted)
    think_closed: str = ""                                       # model | budget | eos | "" (ended inside the block)
    answer_tokens: int = 0
    answer: str | None = None
    finish: str = ""                                             # answer | answer_in_think | no_answer
    done: bool = False

    # filled in by the reward / training code
    confidence: float = float("nan")
    quality: float = float("nan")
    penalty: float = float("nan")
    reward: float = float("nan")
    advantage: float = float("nan")

    # ---- construction helpers -----------------------------------------------------------
    def append_generated(self, ids: list[int], logprobs: np.ndarray) -> None:
        self.response_ids.extend(ids)
        self.response_mask.extend([1] * len(ids))
        self.gen_logprobs.extend(float(x) for x in logprobs)

    def append_inserted(self, ids: list[int]) -> None:
        self.response_ids.extend(ids)
        self.response_mask.extend([0] * len(ids))
        self.gen_logprobs.extend([0.0] * len(ids))

    def drop_last_generated(self, n: int) -> None:
        if n <= 0:
            return
        del self.response_ids[-n:]
        del self.response_mask[-n:]
        del self.gen_logprobs[-n:]

    @property
    def full_ids(self) -> list[int]:
        return self.prompt_ids + self.response_ids

    @property
    def n_generated(self) -> int:
        return int(sum(self.response_mask))

    @property
    def did_search(self) -> bool:
        return self.n_searches > 0

    @property
    def think_truncated(self) -> bool:
        return self.think_closed == "budget"

    def summary(self) -> dict:
        return {"qid": self.qid, "group": self.group, "finish": self.finish, "n_turns": self.n_turns, "n_searches": self.n_searches,
                "think_tokens": self.think_tokens, "think_closed": self.think_closed, "answer_tokens": self.answer_tokens,
                "n_resp": len(self.response_ids), "n_gen": self.n_generated, "answer": self.answer, "gold": self.gold_answer,
                "confidence": self.confidence, "quality": self.quality, "penalty": self.penalty, "reward": self.reward,
                "advantage": self.advantage}


@dataclass
class RolloutStats:
    wall_s: float = 0.0
    generate_s: float = 0.0
    retrieval_s: float = 0.0
    n_retrievals: int = 0
    turns_run: int = 0


def think_budget(cfg: TrainConfig) -> int:
    return int(cfg.max_tokens * cfg.think_fraction)


def answer_budget(cfg: TrainConfig) -> int:
    return cfg.max_tokens - think_budget(cfg)


def new_trajectories(engine: VLLMEngine, items: list[dict], n_per_item: int) -> list[Trajectory]:
    """`items` are HotpotQA records; each becomes a group of n_per_item identical starts."""
    trajs = []
    for g, it in enumerate(items):
        prompt = encode_prompt(engine.tokenizer, it["question"])
        for _ in range(n_per_item):
            trajs.append(Trajectory(qid=it["id"], question=it["question"], gold_answer=it["answer"],
                                    gold_titles=list(it.get("gold_titles", [])), group=g, prompt_ids=list(prompt)))
    return trajs


def run_rollouts(engine: VLLMEngine, retriever: DenseRetriever, trajs: list[Trajectory], cfg: TrainConfig,
                 eval_mode: bool = False, seed: int | None = None, use_lora: bool = True) -> RolloutStats:
    """Drive every trajectory to completion, batching generation and retrieval across trajectories.
    eval_mode uses the EVAL_* sampling settings (Qwen3 advises against greedy decoding while thinking)."""
    tok = engine.tokenizer
    eos_ids = set(_eos_ids(tok))
    close_ids = tok.encode(THINK_CLOSE_FORCED, add_special_tokens=False)
    tb, ab = think_budget(cfg), answer_budget(cfg)
    st = RolloutStats()
    t_start = time.perf_counter()

    def params(i: int, budget: int, stops: list[str], turn: int):
        if eval_mode:
            return engine.eval_params(budget, stops, seed=(cfg.seed * 31 + i * 7919 + turn) % (2 ** 31 - 1))
        s = None if seed is None else (seed * 1_000_003 + i * 7919 + turn * 104_729) % (2 ** 31 - 1)
        return engine.rollout_params(budget, stops, seed=s)

    def close_think(t: Trajectory, how: str) -> None:
        t.append_inserted(close_ids)
        t.think_closed = how
        t.think_tokens = len(t.response_ids)
        t.phase = "answer"

    # ---- phase 1: thinking, searches happen here ----------------------------------------------
    thinking = [i for i, t in enumerate(trajs) if not t.done and t.phase == "think"]
    turn = 0
    while thinking:
        inputs, idxs, sps = [], [], []
        for i in thinking:
            t = trajs[i]
            room = min(tb - len(t.response_ids), cfg.max_model_len - LEN_RESERVE - ab - len(t.full_ids))
            if room < MIN_GEN_TOKENS:
                close_think(t, "budget")
                continue
            inputs.append(t.full_ids)
            idxs.append(i)
            sps.append(params(i, room, THINK_STOPS, turn))
        if idxs:
            t0 = time.perf_counter()
            outs = engine.generate(inputs, sps, use_lora=use_lora)
            st.generate_s += time.perf_counter() - t0
            st.turns_run += 1
            pending: list[tuple[int, str]] = []
            for i, o in zip(idxs, outs):
                t = trajs[i]
                t.n_turns += 1
                t.append_generated(o.token_ids, o.logprobs)
                if o.stop_reason == "</search>":
                    q = last_search_query(o.text)
                    if q:
                        pending.append((i, q))
                        t.searches.append(q)
                    # an empty query inserts nothing; the model just keeps thinking
                elif o.stop_reason == "</think>":
                    t.think_closed = "model"
                    t.think_tokens = len(t.response_ids)
                    t.phase = "answer"
                elif o.stop_reason == "</answer>":                     # answered inside the think block
                    t.answer = extract_answer(o.text)
                    t.think_tokens = len(t.response_ids)
                    t.finish, t.done, t.phase = "answer_in_think", True, "done"
                elif o.finish_reason == "length":
                    close_think(t, "budget")
                else:                                                 # EOS inside the block
                    if o.token_ids and o.token_ids[-1] in eos_ids:
                        t.drop_last_generated(1)
                    close_think(t, "eos")
            if pending:
                t0 = time.perf_counter()
                results = retriever.search([q for _, q in pending], topk=cfg.retrieve_topk)
                st.retrieval_s += time.perf_counter() - t0
                st.n_retrievals += len(pending)
                for (i, _q), passages in zip(pending, results):
                    t = trajs[i]
                    room = tb - len(t.response_ids)
                    if room < MIN_INFO_TOKENS:                        # no budget left for the results
                        close_think(t, "budget")
                        continue
                    t.append_inserted(information_ids(tok, passages, cfg, max_tokens=min(cfg.max_obs_tokens, room)))
                    t.retrieved.append(passages)
                    t.n_searches += 1
        thinking = [i for i in thinking if trajs[i].phase == "think"]
        turn += 1

    # ---- phase 2: the answer ---------------------------------------------------------------------
    inputs, idxs, sps = [], [], []
    for i, t in enumerate(trajs):
        if t.phase != "answer":
            continue
        room = min(ab, cfg.max_model_len - LEN_RESERVE - len(t.full_ids))
        if room < MIN_GEN_TOKENS:
            t.finish, t.done, t.phase = "no_answer", True, "done"
            continue
        inputs.append(t.full_ids)
        idxs.append(i)
        sps.append(params(i, room, ANSWER_STOPS, turn))
    if idxs:
        t0 = time.perf_counter()
        outs = engine.generate(inputs, sps, use_lora=use_lora)
        st.generate_s += time.perf_counter() - t0
        st.turns_run += 1
        for i, o in zip(idxs, outs):
            t = trajs[i]
            t.n_turns += 1
            t.append_generated(o.token_ids, o.logprobs)
            t.answer_tokens = len(o.token_ids)
            t.answer = extract_answer(o.text)
            t.finish = "answer" if t.answer is not None else "no_answer"
            t.done, t.phase = True, "done"

    for t in trajs:
        if not t.done:                                                # cannot happen, but never leave one open
            t.finish, t.done, t.phase = "no_answer", True, "done"
        if not t.text:
            t.text = tok.decode(t.response_ids, skip_special_tokens=False)
    st.wall_s = time.perf_counter() - t_start
    return st


def information_ids(tok, passages: list[Passage], cfg: TrainConfig, max_tokens: int | None = None) -> list[int]:
    """Token ids of the <information> block, with the doc string trimmed so the whole block fits
    `max_tokens` (default MAX_OBS_TOKENS) while keeping the closing tag."""
    limit = cfg.max_obs_tokens if max_tokens is None else max_tokens
    wrapper = len(tok.encode(format_information(""), add_special_tokens=False))
    docs = format_passages(passages, cfg.context_topk)
    doc_ids = tok.encode(docs.strip(), add_special_tokens=False)
    room = max(limit - wrapper, 8)
    if len(doc_ids) > room:
        docs = tok.decode(doc_ids[:room], skip_special_tokens=True)
    return tok.encode(format_information(docs), add_special_tokens=False)


def _eos_ids(tok) -> list[int]:
    ids = []
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    for s in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(s)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.append(i)
    return ids
