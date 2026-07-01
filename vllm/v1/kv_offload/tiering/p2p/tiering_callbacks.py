# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Callback contract from the P2P secondary tier into the TieringManager.

The producer's server role needs three TieringManager-facing operations
to handle an inbound ``LookupMsg`` end-to-end:

1. :meth:`TieringCallbacks.lookup` — ask whether a hash is held locally
   and ready to serve. Returns :class:`LookupResult`.
2. :meth:`TieringCallbacks.create_store_job` — pin the primary-tier
   slots for a set of HIT hashes and obtain a :class:`JobMetadata` whose
   ``job_id`` flows through the existing ``add_stored_blocks`` →
   ``StoreResult`` → ``get_finished_jobs`` → ``complete_read`` path. The
   pin is released by the engine when the JobResult bubbles back.
3. :meth:`TieringCallbacks.finish_request` — release any per-request
   bookkeeping the TieringManager accumulated under the synthetic
   peer-driven :class:`ReqContext`. Called once per inbound LookupMsg
   after all its hashes have settled to HIT or MISS on the wire.

Until the real adapter on :class:`TieringOffloadingManager` lands,
:class:`_AllMissCallbacks` is the production default — every
``lookup`` returns ``MISS``, so ``create_store_job`` is unreachable and
the producer answers all-misses (today's stub behaviour).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.tiering.base import JobMetadata


class TieringCallbacks(Protocol):
    """TieringManager-facing callbacks invoked by the P2P server role."""

    def on_new_request(self, ctx: ReqContext) -> RequestOffloadingContext:
        """Open per-request bookkeeping for the synthetic peer-driven
        ``ctx`` before its first :meth:`lookup`.

        Called once per inbound LookupMsg, before any hash is queried.
        Mirrors :meth:`finish_request`, which releases the same
        bookkeeping once every hash has settled.
        """
        ...

    def lookup(self, key: OffloadKey, ctx: ReqContext) -> LookupResult:
        """Look up a single hash. See :class:`LookupResult` for the
        meaning of each return value."""
        ...

    def create_store_job(
        self,
        keys: Sequence[OffloadKey],
        ctx: ReqContext,
    ) -> JobMetadata:
        """Pin primary-tier slots for ``keys`` and register a store job.

        Caller guarantees every key in ``keys`` is currently HIT (caller
        and callee run on the same scheduler thread, so no eviction can
        race in between). The returned :class:`JobMetadata` has parallel
        ``keys`` and ``block_ids`` of the same length as the input.
        Pin is released when the engine processes the matching
        :class:`JobResult` returned from ``get_finished_jobs``.
        """
        ...

    def finish_request(self, ctx: ReqContext) -> None:
        """Release per-request bookkeeping accumulated under ``ctx``.

        Called once per inbound LookupMsg, after every requested hash
        has resolved to HIT or MISS on the wire (i.e. a definitive
        ``LookupRespMsg`` pair has been sent for it).
        """
        ...


class _AllMissCallbacks:
    """Default callbacks when no real TieringCallbacks is wired.

    Returns MISS for every lookup, which keeps ``create_store_job``
    unreachable in production until the real adapter lands. Preserves
    the all-miss behaviour the server role had before the lookup-phase
    integration.
    """

    def on_new_request(self, ctx: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, ctx: ReqContext) -> LookupResult:
        return LookupResult.MISS

    def create_store_job(
        self,
        keys: Sequence[OffloadKey],
        ctx: ReqContext,
    ) -> JobMetadata:
        raise RuntimeError("create_store_job called without TieringCallbacks wired")

    def finish_request(self, ctx: ReqContext) -> None:
        return
