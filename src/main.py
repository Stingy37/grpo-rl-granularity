"""Entry point: python -m src.main  (run from the repo root; scripts/train_rl.sh does this)

All hyper-parameters come from environment variables (see src/config.py and scripts/train_rl.sh).
A few flags exist for checking pieces without a full run:

  --print-config            show the resolved configuration and exit
  --check-retrieval "q"     run the retriever on one query and print the passages
  --smoke                   tiny end-to-end run (2 groups x 2 rollouts, 2 steps, 4 eval questions)
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print-config", action="store_true")
    ap.add_argument("--check-retrieval", metavar="QUERY")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-eval", action="store_true", help="disable held-out evaluation")
    args = ap.parse_args(argv)

    from src.config import TrainConfig
    cfg = TrainConfig.from_env()
    if args.smoke:
        cfg.num_groups, cfg.rollouts_per_group, cfg.total_steps = 2, 2, 2
        cfg.eval_n, cfg.eval_every, cfg.save_every, cfg.max_tokens = 4, 1, 1, 1536
        cfg.sample_dump_every = 1
        cfg.output_dir = cfg.output_dir.parent / "rl_smoke"
    if args.no_eval:
        cfg.eval_every = 0
    return cfg, args


def main(argv: list[str] | None = None) -> int:
    # quiet, deterministic process environment before anything heavy is imported
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    # FlashInfer's top-k/top-p sampler JIT-compiles with nvcc, which this pod does not have; torch sampling is used
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    cfg, args = parse_args(argv)

    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0

    if args.check_retrieval:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s")
        from src.retriever.dense_retriever import DenseRetriever, format_passages
        r = DenseRetriever(cfg)
        res = r.search([args.check_retrieval])[0]
        print(format_passages(res, cfg.retrieve_topk))
        s = r.last_stats
        print(f"encode {s.encode_ms:.1f} ms | scan {s.scan_ms:.1f} ms ({s.list_bytes / 1e6:.1f} MB of lists) | "
              f"rerank {s.rerank_ms:.1f} ms | text {s.text_ms:.1f} ms")
        return 0

    from src.train_harness.harness import run
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
