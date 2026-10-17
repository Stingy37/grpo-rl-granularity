"""HotpotQA question pools and the per-step assignment of questions to GRPO groups.

Training pool: the HotpotQA train split (90k questions, HF `hotpotqa/hotpot_qa` distractor
parquet shards, or the original JSON/JSONL). Held-out pool: EVAL_N questions drawn once with a
fixed seed from the dev split. If no train split is available, the dev split is shuffled and
split into held-out + train instead. No easy/hard stratification: uniform random sampling
without replacement (reshuffled every epoch) so the corpus distribution comes through.
"""
from __future__ import annotations

import glob
import json
import logging
from pathlib import Path

import numpy as np

from src.config import TrainConfig

log = logging.getLogger(__name__)


def load_hotpot(path: Path) -> list[dict]:
    """-> [{id, question, answer, type, level, gold_titles}] from a parquet file, a directory of
    train-*.parquet shards, a .json (original HotpotQA) or .jsonl file."""
    path = Path(path)
    if path.is_dir():
        files = sorted(glob.glob(str(path / "train-*.parquet"))) or sorted(glob.glob(str(path / "*.parquet")))
        if not files:
            raise FileNotFoundError(f"no parquet shards in {path}")
        out = []
        for f in files:
            out.extend(_load_parquet(Path(f)))
        return out
    if path.suffix == ".parquet":
        return _load_parquet(path)
    if path.suffix == ".jsonl":
        with open(path) as f:
            return [_from_original(json.loads(l)) for l in f if l.strip()]
    if path.suffix == ".json":
        with open(path) as f:
            return [_from_original(r) for r in json.load(f)]
    raise ValueError(f"unsupported HotpotQA file {path}")


def _load_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    tbl = pq.read_table(path, columns=["id", "question", "answer", "type", "level", "supporting_facts"])
    out = []
    for row in tbl.to_pylist():
        gold = list(dict.fromkeys(row["supporting_facts"]["title"]))
        out.append({"id": row["id"], "question": row["question"], "answer": row["answer"], "type": row["type"],
                    "level": row["level"], "gold_titles": gold})
    return out


def _from_original(r: dict) -> dict:
    sf = r.get("supporting_facts", [])
    gold = list(dict.fromkeys(x[0] for x in sf)) if sf and isinstance(sf[0], list) else list(dict.fromkeys(sf.get("title", []))) if isinstance(sf, dict) else []
    return {"id": r.get("_id", r.get("id")), "question": r["question"], "answer": r["answer"], "type": r.get("type", ""),
            "level": r.get("level", ""), "gold_titles": gold}


def build_pools(cfg: TrainConfig) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(cfg.seed)
    dev = load_hotpot(cfg.hotpot_dev_path)
    train_path = Path(cfg.hotpot_train_path) if cfg.hotpot_train_path else None
    if train_path and train_path.exists():
        train = load_hotpot(train_path)
        idx = rng.permutation(len(dev))[:cfg.eval_n]
        heldout = [dev[i] for i in idx]
        log.info("pools: train=%d (from %s), held-out=%d (from dev %d)", len(train), train_path.name, len(heldout), len(dev))
    else:
        idx = rng.permutation(len(dev))
        heldout = [dev[i] for i in idx[:cfg.eval_n]]
        train = [dev[i] for i in idx[cfg.eval_n:]]
        log.warning("no train split at %s: splitting dev into held-out=%d / train=%d", train_path, len(heldout), len(train))
    return train, heldout


class GroupSampler:
    """Uniform sampling without replacement; a new permutation each epoch. `state()` / `load_state()`
    make the sequence resumable."""

    def __init__(self, pool: list[dict], seed: int):
        self.pool = pool
        self.seed = seed
        self.epoch = 0
        self.cursor = 0
        self._perm = self._permutation()

    def _permutation(self) -> np.ndarray:
        return np.random.default_rng(self.seed * 1_000_003 + self.epoch).permutation(len(self.pool))

    def sample(self, n: int) -> list[dict]:
        out = []
        while len(out) < n:
            if self.cursor >= len(self._perm):
                self.epoch += 1
                self.cursor = 0
                self._perm = self._permutation()
            take = self._perm[self.cursor:self.cursor + (n - len(out))]
            out.extend(self.pool[i] for i in take.tolist())
            self.cursor += len(take)
        return out

    def state(self) -> dict:
        return {"epoch": self.epoch, "cursor": self.cursor, "pool_size": len(self.pool)}

    def load_state(self, st: dict) -> None:
        if st.get("pool_size") != len(self.pool):
            log.warning("sampler state is for a pool of %s questions, current pool has %d: restarting the permutation",
                        st.get("pool_size"), len(self.pool))
            return
        self.epoch, self.cursor = int(st["epoch"]), int(st["cursor"])
        self._perm = self._permutation()
