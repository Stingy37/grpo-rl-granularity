"""Dense retrieval over wiki-18 the way Search-R1 does it, under this pod's memory budget.

Search-R1 serves `intfloat/e5-base-v2` + a 64.5 GB IndexFlatIP and returns the top-3 passages
as "Doc i(Title: ...) text". Here the flat index is replaced by the OPQ64,IVF16384,PQ64 index
(54 MB resident) and the flat file is only touched for the exact re-rank:

    query ─e5─> q ─OPQ─> q' ─coarse (16,384 centroids, RAM)─> nprobe lists
          ─ADC scan (lists paged in)─> top-R PQ candidates
          ─pread() R exact rows from e5_Flat.index─> q·x re-score ─> top-k passages ─> parquet text

Measured on HotpotQA-dev (question-only queries): nprobe=128 + re-rank 100 gives recall@10
0.81 vs exact search, any-gold@10 0.77 (exact: 0.82), ~20 ms warm per query.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from dataset.load_index import IVFOPQIndex, load_corpus
from dataset.move_pages import fetch_passages, fetch_vectors_batch, prefetch_lists
from src.config import TrainConfig

log = logging.getLogger(__name__)


@dataclass
class Passage:
    id: int
    title: str
    text: str
    score: float
    exact: bool = True       # False when the score is the PQ approximation (no flat file)


@dataclass
class RetrievalStats:
    n_queries: int = 0
    encode_ms: float = 0.0
    scan_ms: float = 0.0
    rerank_ms: float = 0.0
    text_ms: float = 0.0
    list_bytes: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return self.encode_ms + self.scan_ms + self.rerank_ms + self.text_ms


class E5QueryEncoder:
    """Reproduces how the corpus vectors were made (Search-R1 / FlashRAG conventions):
    "query: " prefix, mean pooling over the attention mask, L2-normalised, max_length 256."""

    def __init__(self, model_dir: str, device: str = "cuda", max_length: int = 256, fp16: bool = True):
        from transformers import AutoModel, AutoTokenizer
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32
        self.model = AutoModel.from_pretrained(model_dir, dtype=dtype).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 128, is_query: bool = True) -> np.ndarray:
        prefix = "query: " if is_query else "passage: "
        out = np.empty((len(texts), self.model.config.hidden_size), dtype=np.float32)
        for i in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[i:i + batch_size]]
            enc = self.tok(batch, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").to(self.device)
            hidden = self.model(**enc, return_dict=True).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * mask).sum(1) / mask.sum(1)
            emb = torch.nn.functional.normalize(emb.float(), dim=-1)
            out[i:i + len(batch)] = emb.cpu().numpy()
        return out

    def to(self, device: str) -> None:
        self.device = device
        self.model.to(device)


class DenseRetriever:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.index, self.store, self.flat = load_corpus(cfg.index_path, cfg.passages_path, cfg.flat_index_path,
                                                        nprobe=cfg.nprobe, nthreads=cfg.faiss_threads)
        self.encoder = E5QueryEncoder(cfg.e5_dir, device=cfg.e5_device)
        self.nprobe = cfg.nprobe
        self.rerank = cfg.rerank_candidates if self.flat is not None else 0
        self.topk = cfg.retrieve_topk
        self.last_stats = RetrievalStats()
        self._self_check()

    def _self_check(self) -> None:
        q = self.encoder.encode(["Were Scott Derrickson and Ed Wood of the same nationality?"] * 50)
        if not self.index.threading_is_consistent(q):
            raise RuntimeError("faiss batch search disagrees with single-threaded search: wrong faiss build "
                               "(pin faiss-cpu==1.13.2) or set FAISS_THREADS=1")
        ps = self.search(["Were Scott Derrickson and Ed Wood of the same nationality?"])[0]
        log.info("retriever ok: %d passages for the probe question, top title %r (%.1f ms)", len(ps), ps[0].title if ps else None,
                 self.last_stats.total_ms)

    # ---- the pipeline -------------------------------------------------------------------
    def search(self, queries: list[str], topk: int | None = None) -> list[list[Passage]]:
        """Batch of query strings -> for each, the top-k passages (best first)."""
        k = topk or self.topk
        st = RetrievalStats(n_queries=len(queries))
        if not queries:
            self.last_stats = st
            return []
        t0 = time.perf_counter()
        q = self.encoder.encode(queries)
        t1 = time.perf_counter()

        # 1-2. rotate, coarse, prefetch the lists, ADC scan -> top-R approximate candidates
        q_rot = self.index.rotate(q)
        list_scores, list_ids = self.index.coarse(q_rot, self.nprobe)
        st.list_bytes = int(self.index.bytes_for_lists(list_ids).sum())
        prefetch_lists(self.index, list_ids)
        n_cand = max(self.rerank, k)
        pq_scores, cand = self.index.scan(q_rot, n_cand, list_ids, list_scores)
        t2 = time.perf_counter()

        # 3. exact re-rank of the candidates from the flat file
        results_ids, results_scores, exact = [], [], self.rerank > 0
        if exact:
            id_lists = [c[c >= 0] for c in cand]
            vecs = fetch_vectors_batch(self.flat, id_lists, threads=self.cfg.rerank_threads)
            for i in range(len(queries)):
                s = vecs[i] @ q[i]
                top = np.argsort(-s)[:k]
                results_ids.append(id_lists[i][top])
                results_scores.append(s[top])
        else:
            for i in range(len(queries)):
                keep = cand[i] >= 0
                results_ids.append(cand[i][keep][:k])
                results_scores.append(pq_scores[i][keep][:k])
        t3 = time.perf_counter()

        # 4. text
        flat_ids = np.concatenate(results_ids) if results_ids else np.empty(0, np.int64)
        rows = fetch_passages(self.store, flat_ids)
        out, pos = [], 0
        for ids, scores in zip(results_ids, results_scores):
            ps = []
            for j in range(len(ids)):
                r = rows[pos + j]
                ps.append(Passage(id=int(r["id"]), title=r["title"], text=r["text"], score=float(scores[j]), exact=exact))
            pos += len(ids)
            out.append(ps)
        t4 = time.perf_counter()

        st.encode_ms, st.scan_ms, st.rerank_ms, st.text_ms = (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3
        self.last_stats = st
        return out

    def close(self) -> None:
        self.index.close()
        if self.flat is not None:
            self.flat.close()


# ----------------------------------------------------------------------------------------------
# helpers shared by the rollout loop and the evaluator
# ----------------------------------------------------------------------------------------------
def format_passages(passages: list[Passage], k: int) -> str:
    """Search-R1's _passages2string: 'Doc i(Title: title) text\\n' for the top-k."""
    return "".join(f"Doc {i + 1}(Title: {p.title}) {p.text}\n" for i, p in enumerate(passages[:k]))


def _norm_title(t: str) -> str:
    return " ".join(t.lower().replace("_", " ").split())


def gold_hits(gold_titles: list[str], passages: list[Passage]) -> tuple[int, int]:
    """(#gold titles found among passages, #gold titles). Titles compared case/space-insensitively."""
    seen = {_norm_title(p.title) for p in passages}
    gold = {_norm_title(t) for t in gold_titles}
    return len(gold & seen), len(gold)
