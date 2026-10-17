"""Moving corpus bytes from disk into RAM on demand, and measuring how much moved.

The index, the flat vectors and the passages all live on NFS and are far larger than the
RAM this pod is allowed, so every fetch here is explicit about what it touches:

  prefetch_lists   posix_fadvise(WILLNEED) on the byte ranges of the inverted lists a query
                   is about to scan, so the kernel pulls them while the CPU is still busy
                   with the coarse step (cold NFS reads are ~2.6 ms per list, 128 lists per query)
  fetch_vectors    exact fp32 rows of the re-rank candidates: R random 3 KB pread()s, threaded,
                   because latency rather than bandwidth is what an NFS random read costs
  fetch_passages   text of the final top-k from the parquet row groups that contain them
  resident_set / file_resident_bytes / drop_file_cache
                   accounting helpers: what the process owns, what of a file is in page cache,
                   and how to evict it (to measure a cold query without root)
"""
from __future__ import annotations

import ctypes
import mmap
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from dataset.load_index import FlatVectors, IVFOPQIndex, PassageStore


# ----------------------------------------------------------------------------------------------
# inverted lists (the IVF scan)
# ----------------------------------------------------------------------------------------------
def prefetch_lists(index: IVFOPQIndex, list_ids: np.ndarray) -> int:
    """Ask the kernel to start reading the probed lists. Returns bytes requested (0 if the index
    is not memory-mapped from disk, in which case nothing needs moving)."""
    if index.list_offsets is None:
        return 0
    ids = np.unique(np.asarray(list_ids).ravel())
    ids = ids[ids >= 0]
    fd = os.open(index.path, os.O_RDONLY)
    total = 0
    try:
        for l in ids.tolist():
            nbytes = int(index.list_sizes[l]) * index.bytes_per_entry
            if nbytes:
                os.posix_fadvise(fd, int(index.list_offsets[l]), nbytes, os.POSIX_FADV_WILLNEED)
                total += nbytes
    finally:
        os.close(fd)
    return total


# ----------------------------------------------------------------------------------------------
# exact vectors (the re-rank)
# ----------------------------------------------------------------------------------------------
def fetch_vectors(flat: FlatVectors, ids: np.ndarray, threads: int = 32) -> np.ndarray:
    """Exact fp32 rows for `ids` from the flat file. One pread() per row, in parallel."""
    ids = np.asarray(ids, dtype=np.int64)
    out = np.empty((len(ids), flat.d), dtype=np.float32)
    if len(ids) == 0:
        return out

    def read_one(j: int) -> None:
        flat.read_rows(int(ids[j]), int(ids[j]) + 1, out[j:j + 1])

    with ThreadPoolExecutor(max_workers=min(threads, len(ids))) as ex:
        list(ex.map(read_one, range(len(ids))))
    return out


def fetch_vectors_batch(flat: FlatVectors, id_lists: list[np.ndarray], threads: int = 32) -> list[np.ndarray]:
    """Same as fetch_vectors for several queries at once, sharing one thread pool so the NFS
    round trips of all queries overlap. Duplicate ids across queries are read once."""
    uniq = np.unique(np.concatenate([np.asarray(x, dtype=np.int64) for x in id_lists])) if id_lists else np.empty(0, np.int64)
    vecs = fetch_vectors(flat, uniq, threads)
    pos = {int(i): j for j, i in enumerate(uniq.tolist())}
    return [vecs[[pos[int(i)] for i in ids]] if len(ids) else np.empty((0, flat.d), np.float32) for ids in id_lists]


# ----------------------------------------------------------------------------------------------
# passage text
# ----------------------------------------------------------------------------------------------
def fetch_passages(store: PassageStore, ids) -> list[dict]:
    return store.get(ids)


# ----------------------------------------------------------------------------------------------
# accounting
# ----------------------------------------------------------------------------------------------
def resident_set() -> dict:
    """RssAnon = private memory this process owns; RssFile = mapped file pages currently resident
    (reclaimable). Bytes."""
    with open("/proc/self/status") as f:
        s = f.read()
    return {k: int(re.search(rf"{k}:\s+(\d+)", s).group(1)) * 1024 for k in ("RssAnon", "RssFile")}


def drop_file_cache(path: Path) -> None:
    """Evict a file's clean pages from the page cache (no root). Only pages not currently mapped
    by any process are dropped, so close the index first to measure a truly cold query."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def file_resident_bytes(path: Path) -> int | None:
    """How many bytes of `path` are in the page cache right now (mincore)."""
    size = os.path.getsize(path)
    if size == 0:
        return 0
    try:
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE) as mm:
            pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
            vec = (ctypes.c_ubyte * pages)()
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
                return None
            resident = np.count_nonzero(np.frombuffer(vec, dtype=np.uint8) & 1)
            del vec
        return int(min(resident * mmap.PAGESIZE, size))
    except Exception:
        return None
