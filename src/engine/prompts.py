"""Prompt format for the policy: Search-R1's instruction, adapted to Qwen3's native thinking.

Search-R1 asked its (non-thinking) Qwen2.5 / Llama models to write <think> ... </think> as plain
text and interleaved <search> / <information> with it. Qwen3 has a native thinking block that uses
the very same <think> / </think> tokens, so here that block *is* the reasoning channel: the prompt
ends with "<think>\\n" (forced open), the model reasons, searches with <search> query </search>
and receives "\\n\\n<information>Doc 1(Title: ...) ...</information>\\n\\n" spliced into the block
(those tokens are masked out of the loss, as Search-R1's state_masking does), closes the block
with </think> and answers inside <answer> ... </answer>. No SFT: the instruction below is all the
model is told. rollout.py enforces the token budgets (thinking / answer) and inserts a forced
"\\n</think>\\n\\n" when the thinking budget is used up.
"""
from __future__ import annotations

import re

# Search-R1 scripts/data_process/*_search.py make_prefix(), reworded for one native thinking block.
SEARCH_R1_TEMPLATE = (
    "Answer the given question. "
    "Think step by step inside <think> and </think>. "
    "While thinking, whenever you find you lack some knowledge, you can call a search engine by writing "
    "<search> query </search>; the top searched results will be inserted right after it between <information> "
    "and </information>, and you can keep reasoning with them. "
    "You can search as many times as you want. "
    "When no further external knowledge is needed, close your thinking with </think> and directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
    "Question: {question}\n"
)

# Search-R1 llm_agent/generation.py
INFORMATION_TEMPLATE = "\n\n<information>{}</information>\n\n"

THINK_OPEN = "<think>\n"                  # appended to the prompt: the block is always open
THINK_CLOSE_FORCED = "\n</think>\n\n"     # inserted when the thinking budget runs out (Qwen3's native closing)
THINK_STOPS = ["</search>", "</think>", "</answer>"]
ANSWER_STOPS = ["</answer>"]

_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def question_text(q: str) -> str:
    q = q.strip()
    return q if q.endswith("?") else q + "?"


def build_messages(question: str) -> list[dict]:
    """Search-R1 puts the whole instruction in a single user message, no system prompt."""
    return [{"role": "user", "content": SEARCH_R1_TEMPLATE.format(question=question_text(question))}]


def encode_prompt(tokenizer, question: str) -> list[int]:
    """Token ids of the chat-formatted prompt, assistant header included, with the thinking block
    forced open (Qwen3's template with thinking enabled leaves the model to emit "<think>" itself)."""
    msgs = build_messages(question)
    try:
        ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=True)
    except TypeError:
        ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "input_ids") or isinstance(ids, dict):          # transformers >= 5 returns a BatchEncoding
        ids = ids["input_ids"]
    return list(ids) + tokenizer.encode(THINK_OPEN, add_special_tokens=False)


def last_search_query(text: str) -> str:
    """Query of the last closed <search> tag in a generated chunk (generation stops at the first
    </search>, so there is at most one)."""
    ms = _SEARCH_RE.findall(text)
    return ms[-1].strip() if ms else ""


def extract_answer(text: str) -> str | None:
    """Last <answer> block in `text`, or None."""
    ms = _ANSWER_RE.findall(text)
    return ms[-1].strip() if ms else None


def extract_searches(text: str) -> list[str]:
    return [m.strip() for m in _SEARCH_RE.findall(text)]


def split_think(response: str) -> tuple[str, str]:
    """(thinking text, text after </think>) of a decoded response; the whole response counts as
    thinking when the block was never closed."""
    head, sep, tail = response.partition("</think>")
    return (head, tail) if sep else (response, "")


def format_information(doc_string: str) -> str:
    return INFORMATION_TEMPLATE.format(doc_string.strip())
