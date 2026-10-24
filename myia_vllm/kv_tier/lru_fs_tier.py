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
"""

import os
import time
from collections.abc import Collection

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

logger = init_logger(__name__)

_ENABLED = os.environ.get("MYIA_FS_TOUCH", "1") != "0"
_MAX_WARNINGS = 5


class TouchOnHitFsTier(FileSystemTierManager):
    """``FileSystemTierManager`` that records use, not just writes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ok = 0
        self._miss = 0
        self._err = 0
        self._warned = 0
        if not _ENABLED:
            logger.warning("TouchOnHitFsTier: MYIA_FS_TOUCH=0 — recency tracking disabled")

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
                }
            except Exception:
                pass
        return stats
