"""Open the three on-disk pieces of the wiki-18 corpus without loading them into RAM.

  IVFOPQIndex    faiss OPQ64,IVF16384,PQ64 over the 21,015,324 e5-base-v2 passage vectors,
                 opened with IO_FLAG_MMAP: the ~54 MB header (coarse centroids, PQ codebooks,
                 OPQ rotation, list offsets) is read into RAM, the 1.5 GB of codes+ids stays
                 on disk and a search page-faults in exactly the `nprobe` lists it scans.
  FlatVectors    Search-R1's e5_Flat.index (64.5 GB IndexFlatIP = 45-byte header + raw fp32
                 rows). Never mapped; single rows are pread() for the exact re-rank step.
  PassageStore   wiki18_passages.parquet (id/title/text, 8192-row zstd row groups). A fetch
                 decodes only the row groups holding the requested ids.

Memory-lean retrieval (see src/retriever/dense_retriever.py):
  1. rotate the query with the OPQ matrix, score it against the 16,384 coarse centroids (RAM)
  2. ADC-scan the top-`nprobe` inverted lists (paged in, ~12 MB at nprobe=128)  -> top-R codes
  3. pread() the R exact 3 KB rows from the flat file, re-score, keep top-k      -> passages
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.config import DIM, FLAT_HEADER_BYTES, N_PASSAGES

log = logging.getLogger(__name__)

ID_BYTES = 8
GOOD_FAISS = "1.13.2"


def check_faiss() -> None:
    """faiss-cpu 1.15.1 silently corrupts batched IndexIVF results with >= 8 OpenMP threads
    (top-10 overlap with single-thread search ~5%). 1.13.2 was verified correct on this corpus;
    IVFOPQIndex.threading_is_consistent() re-checks at startup whatever the version."""
    import faiss
    if faiss.__version__ != GOOD_FAISS:
        log.warning("faiss %s != %s (the version validated on this index); relying on the runtime consistency check",
                    faiss.__version__, GOOD_FAISS)


class IVFOPQIndex:
    def __init__(self, path: Path, nprobe: int = 128, nthreads: int = 16, mmap: bool = True):
        import faiss
        check_faiss()
        self.path = Path(path)
        faiss.omp_set_num_threads(nthreads)
        flags = (faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY) if mmap else 0
        t0 = time.time()
        self.index = faiss.read_index(str(self.path), flags)
        self.load_seconds = time.time() - t0
        self.ivf = faiss.downcast_index(faiss.extract_index_ivf(self.index))
        self.ivf.nprobe = int(nprobe)
        # the OPQ variant wraps the IVFPQ in an IndexPreTransform whose rotation precedes the coarse step
        pre = faiss.downcast_index(self.index)
        self.transform = faiss.downcast_VectorTransform(pre.chain.at(0)) if isinstance(pre, faiss.IndexPreTransform) else None
        self.d, self.nlist, self.code_size = self.index.d, self.ivf.nlist, self.ivf.code_size
        self.bytes_per_entry = self.code_size + ID_BYTES
        self.list_sizes = np.array([self.ivf.invlists.list_size(i) for i in range(self.nlist)], dtype=np.int64)
        # byte offsets of each inverted list inside the file (codes then ids), for prefetching (move_pages.py)
        self.list_offsets: np.ndarray | None = None
        try:
            od = faiss.downcast_InvertedLists(self.ivf.invlists)
            self.list_offsets = np.array([od.lists.at(i).offset for i in range(self.nlist)], dtype=np.int64)
        except Exception:  # not an OnDiskInvertedLists (mmap=False); prefetch is then a no-op
            pass
        log.info("index %s: nlist=%d code=%dB nprobe=%d mmap=%s opened in %.1fs",
                 self.path.name, self.nlist, self.code_size, nprobe, mmap, self.load_seconds)

    # ---- query-side ---------------------------------------------------------------------
    @property
    def nprobe(self) -> int:
        return self.ivf.nprobe

    @nprobe.setter
    def nprobe(self, v: int) -> None:
        self.ivf.nprobe = int(v)

    def rotate(self, q: np.ndarray) -> np.ndarray:
        """Apply the OPQ rotation (identity for a plain IVFPQ index)."""
        q = np.ascontiguousarray(q, dtype=np.float32).reshape(-1, self.d)
        return self.transform.apply_py(q) if self.transform is not None else q

    def coarse(self, q: np.ndarray, nprobe: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Step 1: (centroid scores, list ids) of the nprobe nearest coarse centroids. q is already rotated."""
        return self.ivf.quantizer.search(np.ascontiguousarray(q, dtype=np.float32), nprobe or self.nprobe)

    def scan(self, q_rot: np.ndarray, k: int, list_ids: np.ndarray, list_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Step 2: ADC scan of the given lists for an already-rotated query batch."""
        self.ivf.nprobe = list_ids.shape[1]
        return self.ivf.search_preassigned(np.ascontiguousarray(q_rot, dtype=np.float32), k, list_ids, list_scores)

    def search(self, q: np.ndarray, k: int, nprobe: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Steps 1+2 in one faiss call (rotation included). Scores are approximate inner products."""
        if nprobe is not None:
            self.ivf.nprobe = int(nprobe)
        q = np.ascontiguousarray(q, dtype=np.float32).reshape(-1, self.d)
        return self.index.search(q, k)

    def bytes_for_lists(self, list_ids: np.ndarray) -> np.ndarray:
        return self.list_sizes[list_ids].sum(axis=-1) * self.bytes_per_entry

    def threading_is_consistent(self, q: np.ndarray, k: int = 10) -> bool:
        """Multi-threaded batch search must equal single-threaded search (guards the 1.15.1 bug)."""
        import faiss
        q = np.ascontiguousarray(q[:50], dtype=np.float32)
        nt = faiss.omp_get_max_threads()
        _, batch = self.search(q, k)
        faiss.omp_set_num_threads(1)
        _, single = self.search(q, k)
        faiss.omp_set_num_threads(nt)
        return bool(np.array_equal(batch, single))

    def close(self) -> None:
        self.transform = None
        self.ivf = None
        self.index = None


class FlatVectors:
    """pread() access to rows of e5_Flat.index. Only the rows asked for ever touch RAM."""

    def __init__(self, path: Path, drop_cache: bool = False):
        self.path = Path(path)
        self.n, self.d = N_PASSAGES, DIM
        self.row_bytes = self.d * 4
        expected = FLAT_HEADER_BYTES + self.n * self.row_bytes
        actual = os.path.getsize(self.path)
        if actual != expected:
            raise ValueError(f"{self.path}: {actual} bytes, expected {expected} (45-byte header + {self.n}x{self.d} fp32)")
        self.fd = os.open(self.path, os.O_RDONLY)
        self.drop_cache = drop_cache

    def read_rows(self, start: int, stop: int, out: np.ndarray | None = None) -> np.ndarray:
        n = stop - start
        buf = out[:n] if out is not None else np.empty((n, self.d), dtype=np.float32)
        mv = memoryview(buf).cast("B")
        nbytes, off, done = n * self.row_bytes, FLAT_HEADER_BYTES + start * self.row_bytes, 0
        while done < nbytes:
            got = os.preadv(self.fd, [mv[done:]], off + done)
            if got <= 0:
                raise EOFError(f"short read at row {start}: {done}/{nbytes} bytes")
            done += got
        if self.drop_cache:
            os.posix_fadvise(self.fd, off, nbytes, os.POSIX_FADV_DONTNEED)
        return buf

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def split_contents(contents: str) -> tuple[str, str]:
    """wiki-18 stores each passage as '"Title"\\ntext'."""
    head, _, text = contents.partition("\n")
    if len(head) >= 2 and head[0] == '"' and head[-1] == '"':
        head = head[1:-1]
    return head, text


class PassageStore:
    """Random access to passage text by corpus row id via parquet row groups."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.pf = pq.ParquetFile(self.path)
        self.row_group = self.pf.metadata.row_group(0).num_rows
        self.n = self.pf.metadata.num_rows
        if self.n != N_PASSAGES:
            raise ValueError(f"{self.path}: {self.n} rows, expected {N_PASSAGES}")

    def get(self, ids) -> list[dict]:
        ids = [int(i) for i in ids]
        by_group: dict[int, list[int]] = {}
        for i in ids:
            by_group.setdefault(i // self.row_group, []).append(i)
        found: dict[int, dict] = {}
        for g, members in by_group.items():
            tbl = self.pf.read_row_group(g)
            base = g * self.row_group
            for i in members:
                found[i] = tbl.slice(i - base, 1).to_pylist()[0]
        return [found[i] for i in ids]


def load_corpus(index_path: Path, passages_path: Path, flat_path: Path | None, nprobe: int, nthreads: int):
    """Open everything the retriever needs. Returns (IVFOPQIndex, PassageStore, FlatVectors | None)."""
    index = IVFOPQIndex(index_path, nprobe=nprobe, nthreads=nthreads)
    store = PassageStore(passages_path)
    flat = FlatVectors(flat_path) if flat_path and Path(flat_path).exists() else None
    if flat is None:
        log.warning("flat index not available: retrieval will return PQ-approximate scores without exact re-ranking")
    return index, store, flat
