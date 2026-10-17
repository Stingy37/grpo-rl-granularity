"""Low-level logging: the text log, the per-step metrics JSONL and trajectory sample dumps.

output/rl/logs/
  train_<timestamp>.log     everything the stdlib loggers emit (also mirrored to the console)
  metrics.jsonl             one JSON record per GRPO step (what dashboard.py prints, plus more)
  eval.jsonl                one record per held-out evaluation
  samples/step_NNNNN.jsonl  a few complete trajectories (best / worst / random) for reading
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Writes to the log file only. The dashboard prints its table to the console itself and mirrors
# the lines here, so they appear once on screen and once in the file.
FILE_ONLY_LOGGER = "grpo.file"


def setup_logging(log_dir: Path, level: str = "INFO") -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname).1s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    fl = logging.getLogger(FILE_ONLY_LOGGER)
    for h in list(fl.handlers):
        fl.removeHandler(h)
    fl.addHandler(fh)
    fl.propagate = False
    fl.setLevel(logging.INFO)
    for noisy in ("vllm", "httpx", "urllib3", "filelock", "transformers", "peft", "faiss", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path


class MetricsLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.samples_dir = self.log_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self.eval_path = self.log_dir / "eval.jsonl"

    def log_step(self, record: dict) -> None:
        self._append(self.metrics_path, record)

    def log_eval(self, record: dict) -> None:
        self._append(self.eval_path, record)

    def history(self) -> list[dict]:
        if not self.metrics_path.exists():
            return []
        out = []
        with open(self.metrics_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return out

    def dump_samples(self, step: int, trajs: list, n_random: int = 2) -> Path:
        """Best, worst and a couple of random trajectories of this step, in full."""
        if not trajs:
            return self.samples_dir
        order = sorted(range(len(trajs)), key=lambda i: trajs[i].reward if trajs[i].reward == trajs[i].reward else -1e9)
        pick = {order[-1]: "best", order[0]: "worst"}
        rng = np.random.default_rng(step)
        for i in rng.choice(len(trajs), size=min(n_random, len(trajs)), replace=False).tolist():
            pick.setdefault(i, "random")
        path = self.samples_dir / f"step_{step:05d}.jsonl"
        with open(path, "w") as f:
            for i, tag in pick.items():
                t = trajs[i]
                rec = {"tag": tag, "step": step, **t.summary(), "question": t.question, "searches": t.searches,
                       "retrieved_titles": [[p.title for p in ps] for ps in t.retrieved], "text": t.text}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path

    @staticmethod
    def _append(path: Path, record: dict) -> None:
        record = {"ts": time.time(), **record}
        with open(path, "a") as f:
            f.write(json.dumps(_jsonable(record)) + "\n")
            f.flush()
            os.fsync(f.fileno())


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and (x != x):
        return None
    return x
