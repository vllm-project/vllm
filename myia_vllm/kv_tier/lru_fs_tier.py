# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree KV offloading tier: filesystem, with mtime = last *use*.

The problem this solves
-----------------------
Our external pruner (``scripts/monitoring/fstier_pruner.py``) deletes the
oldest block files by mtime. Upstream writes a block once and never touches it
again, so mtime means "last written". For a prefix cache that is the wrong
ordering: a 200K-token conversation prefix is written once and read on every
subsequent turn, so the oldest-by-write blocks are often the most valuable.

What upstream gives us, and what actually works
-----------------------------------------------
``SecondaryTierManager`` declares ``touch(keys, req_context)`` — "Mark chunks
as recently used for eviction policy" — and ``TieringOffloadingManager.touch``
forwards it to every secondary tier. **In v0.32.0 that hook is dead**: a full
grep of the installed package finds no caller of the offloading manager's
``touch``; only the GPU block pool, the LoRA adapters, the mooncake connector
and ``simple_kv_offload`` use their own unrelated ``touch``. Wiring a tier on
``touch`` would produce a no-op that looks installed.

The live hook is ``lookup()``. ``tiering/manager.py:454`` calls
``tier.lookup(key, req_context)`` for every secondary tier on every key it
probes, and times it. ``FileSystemTierManager.lookup`` returns
``LookupResult.HIT`` when the block file exists and ``MISS``/``RETRY``
otherwise. A ``HIT`` is precisely "this block is in the tier and is about to
be reused", which is the recency signal the pruner needs.

So: on HIT, bump the block file's mtime. The external pruner, unchanged,
becomes an LRU over reads. No journal, no side index, and recency survives
process and host restarts because it lives in the filesystem metadata.

Why LRU is the right policy for a prefix tree
---------------------------------------------
Blocks are chained by hash, so a conversation is a chain of block files.
Deleting a shared root invalidates every descendant; deleting a leaf
invalidates one tail. Under LRU-by-use the shared roots are hit by every turn
of every conversation that shares them, so they are the most recently used
blocks in the tier — the policy protects exactly the blocks whose loss costs
most, without modelling the tree.

Cost
----
One ``utime`` syscall per HIT. Several hundred per minute at the observed
request rate, against a tier that already does thousands of ``faccessat``
lookups per minute through the ``fs_io_C`` batch path. Over-touch is possible
but harmless: ``lookup`` may be probed for blocks a request ends up not
loading, which only makes a block look slightly more recent than it was.

Deployment
----------
Referenced from the connector config; ``SecondaryTierFactory.get_tier_class``
imports the class when ``type`` is not a registered name::

    {"type": "TouchOnHitFsTier",
     "module_path": "lru_fs_tier",
     "root_dir": "/kv-tier-fs", "n_read_threads": 16, "n_write_threads": 16}

Kill switch without touching the connector config: ``MYIA_FS_TOUCH=0`` in the
container environment makes every touch a no-op, restoring exact upstream
behaviour. The variable is read once, at module import, so it takes effect only
when the container is recreated (an env change needs a recreate anyway); it is
not a live toggle.

Write admission (``MYIA_FS_SKIP_REWRITE``, default on)
-----------------------------------------------------
A second, independent job: ``submit_store`` drops any block whose file already
exists on disk. Content-addressed filenames plus upstream's atomic
``.tmp`` + ``os.replace()`` mean a present file *is* the block, so the rewrite
only copies identical bytes over identical bytes. Measured on production:
51 % of fs write volume was exactly that. ``MYIA_FS_SKIP_REWRITE=0`` restores
upstream write-through (also read at import, also needs a recreate).

The two switches are independent: ``MYIA_FS_TOUCH`` governs recency tracking on
HIT, ``MYIA_FS_SKIP_REWRITE`` governs write volume. Setting the second to 0
without the first still leaves correct LRU behaviour.
"""

import dataclasses
import os
import time
from collections.abc import Collection

try:  # same C fast path upstream uses for its own existence checks
    from vllm.fs_io_C import batch_lookup as _batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:  # pragma: no cover - falls back to os.path.exists
    _batch_lookup_C = None
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

logger = init_logger(__name__)

_ENABLED = os.environ.get("MYIA_FS_TOUCH", "1") != "0"
_SKIP_REWRITE = os.environ.get("MYIA_FS_SKIP_REWRITE", "1") != "0"
_MAX_WARNINGS = 5


def _noop_store() -> None:
    """Stand-in task for a store job whose blocks are all already on disk.

    The job must still reach the thread pool: ``TieringOffloadingManager``
    keeps per-job state (``_jobs``, ``_job_block_counts``) and only clears it
    when ``get_finished_jobs()`` reports the job, so a job dropped silently
    would leak that state for the life of the process.
    """


class TouchOnHitFsTier(FileSystemTierManager):
    """``FileSystemTierManager`` that records use, not just writes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ok = 0
        self._miss = 0
        self._err = 0
        self._warned = 0
        self._skip_rewrite = _SKIP_REWRITE
        self._skipped_keys = 0
        self._written_keys = 0
        if not _ENABLED:
            logger.warning("TouchOnHitFsTier: MYIA_FS_TOUCH=0 — recency tracking disabled")
        if not _SKIP_REWRITE:
            logger.warning(
                "TouchOnHitFsTier: MYIA_FS_SKIP_REWRITE=0 — every store rewrites "
                "its blocks, upstream write-through behaviour restored"
            )

    def _mark_used(self, key: OffloadKey) -> None:
        try:
            os.utime(self.file_mapper.get_file_name(key), None)
            self._ok += 1
        except FileNotFoundError:
            # Raced with a prune between the lookup and the utime, or the block
            # lives only in the primary tier. Not an error, and never raised.
            self._miss += 1
        except OSError as exc:
            self._err += 1
            if self._warned < _MAX_WARNINGS:
                self._warned += 1
                logger.warning(
                    "TouchOnHitFsTier: utime failed (%s); %d warning(s) max per process",
                    exc,
                    _MAX_WARNINGS,
                )

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = super().lookup(key, req_context)
        if _ENABLED and result is LookupResult.HIT:
            self._mark_used(key)
        return result

    @staticmethod
    def _existing_flags(paths: list[str]) -> list[bool]:
        if _HAS_BATCH_LOOKUP_C:
            return list(_batch_lookup_C(paths))
        return [os.path.exists(p) for p in paths]

    @override
    def submit_store(self, job_metadata) -> None:
        """Store only the blocks whose file is not already on disk.

        Why this is safe by construction, not by inspection: block filenames are
        content-addressed (a hash of the token-block chain, under a per-model
        namespace and rank), and upstream writes each file through
        ``<dest>.tmp`` + ``os.replace()``, which is atomic. A file present at
        its final path is therefore a *complete* file holding exactly the bytes
        this job would write. Re-writing it copies identical content over
        identical content — pure write amplification against the SSD.

        Measured on production before this change: 51 % of the fs write volume
        was such re-writes (154 of 303 files in a 30 min window, 4.04 GB of
        7.95 GB). See ``_iteration_log/v0320_fstier/NOTES.md`` 26/10 00:15Z.

        Failure mode if the probe lies: the block is not written, so a later
        lookup MISSes and the prefix is recomputed. There is no correctness
        hazard — a rejected store only costs prefill, never output.
        """
        if not self._skip_rewrite:
            return super().submit_store(job_metadata)

        keys = list(job_metadata.keys)
        if not keys:
            return super().submit_store(job_metadata)

        paths = [self.file_mapper.get_file_name(k) for k in keys]
        try:
            existing = self._existing_flags(paths)
        except Exception as exc:  # fail open: never lose a store to this probe
            self._err += 1
            if self._warned < _MAX_WARNINGS:
                self._warned += 1
                logger.warning(
                    "TouchOnHitFsTier: existence probe failed (%s); storing all "
                    "%d blocks as upstream would",
                    exc,
                    len(keys),
                )
            return super().submit_store(job_metadata)

        keep = [i for i, present in enumerate(existing) if not present]
        skipped = len(keys) - len(keep)
        self._skipped_keys += skipped
        self._written_keys += len(keep)

        if not keep:
            # Every block already on disk. The job still has to *finish* on the
            # pool or the manager never clears its per-job state, so enqueue a
            # task that does nothing. _store_job_keys is kept: the blocks are
            # in the tier, and the manager derives its "this key is now
            # offloaded" event from it.
            if self.events is not None:
                self._store_job_keys[job_metadata.job_id] = keys
            self._job_block_counts[job_metadata.job_id] = 0
            self._pool.enqueue_store(job_metadata.job_id, 1, [_noop_store])
            return

        if not skipped:
            return super().submit_store(job_metadata)

        # Partial: hand upstream a job carrying only the missing blocks,
        # keeping job_id so the manager's per-job state stays consistent.
        chunk_ids = job_metadata.chunk_ids
        super().submit_store(
            dataclasses.replace(
                job_metadata,
                keys=[keys[i] for i in keep],
                chunk_ids=chunk_ids[keep] if chunk_ids is not None else None,
            )
        )

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        """Kept for forward compatibility: dead in v0.32.0, harmless if wired."""
        if not (_ENABLED and keys):
            return
        for key in keys:
            self._mark_used(key)

    def get_stats(self):  # pragma: no cover - observability only
        stats = super().get_stats()
        if stats is not None:
            try:
                stats.metadata = {
                    **(getattr(stats, "metadata", {}) or {}),
                    "touch_ok": self._ok,
                    "touch_miss": self._miss,
                    "touch_err": self._err,
                    "store_written_keys": self._written_keys,
                    "store_skipped_keys": self._skipped_keys,
                }
            except Exception:
                pass
        return stats
