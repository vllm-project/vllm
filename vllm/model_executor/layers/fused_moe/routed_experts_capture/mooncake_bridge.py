# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side bridge: routed-experts routing over the Mooncake KV pool.

The ``MooncakeStoreConnector`` moves KV blocks directly between GPU and a
``MooncakeDistributedStore`` (RDMA pool) in the worker process — no CPU offload
tier. This bridge makes the per-token routing follow the SAME blocks on the
SAME transport: when the connector PUTs a KV block it also PUTs that block's
routing row under ``"re:"+key``; when it GETs a KV block back (cache hit) it
GETs the routing row straight into the scheduler-shared ``/dev/shm`` slot
buffer, where the scheduler reads it via ``RoutedExpertsManager.get`` exactly as
for the on-GPU path.

Only the ``output_rank`` worker runs a bridge (routing is the global top-k,
replicated across TP ranks; the shared slot buffer is per-DP-rank and only
``output_rank`` attaches it). The routing row of block ``b`` is the contiguous
``block_row_nbytes`` at ``base + b * block_row_nbytes`` of the slot buffer —
the same block-offset addressing the Mooncake KV path uses — so PUT/GET reuse
``batch_put_from_multi_buffers`` / ``batch_get_into_multi_buffers`` with a single
segment per key. NO FALLBACK: any negative return code raises.
"""

from __future__ import annotations

from collections.abc import Sequence

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capture.shared_region import (
    RoutedExpertsWorkerWriter,
)

logger = init_logger(__name__)

_ROUTING_KEY_PREFIX = "re:"


class RoutedExpertsMooncakeBridge:
    """PUT/GET routing rows alongside KV blocks on the Mooncake pool.

    Wraps the ``output_rank`` worker's ``RoutedExpertsWorkerWriter`` (which owns
    the shared slot-buffer mmap) and the connector's ``MooncakeDistributedStore``
    handle. The slot buffer is registered with the store once at
    ``register_kv_caches`` time so reads/writes are zero-copy RDMA into the same
    buffer the scheduler reads.
    """

    def __init__(
        self, store, writer: RoutedExpertsWorkerWriter, replicate_config=None
    ) -> None:
        self._store = store
        self._writer = writer
        # Same ReplicateConfig the connector's KV PUT uses, so routing rows get
        # the SAME replica count / preferred-segment placement as the KV blocks
        # they ride with. None -> store default (mirrors a connector with no
        # replicate config).
        self._replicate_config = replicate_config
        self._block_bytes = writer.block_row_nbytes()
        self._registered = False

    def register(self) -> None:
        """Attach the slot buffer and register it with Mooncake (idempotent)."""
        if self._registered:
            return
        self._writer.attach()
        base = self._writer.region_base_address()
        nbytes = self._writer.region_nbytes()
        ret = self._store.register_buffer(base, nbytes)
        if ret != 0:
            raise RuntimeError(
                f"Mooncake register_buffer for routing slot buffer failed "
                f"(addr={base:#x}, len={nbytes}, ret={ret})"
            )
        self._base = base
        self._registered = True
        logger.info(
            "Registered routed-experts slot buffer with Mooncake "
            "(%.2f GB, block_row=%d bytes)",
            nbytes / 1e9,
            self._block_bytes,
        )

    def _routing_addrs_sizes(
        self, block_ids: Sequence[int]
    ) -> tuple[list[list[int]], list[list[int]]]:
        """One contiguous segment per block: the block's routing row."""
        addrs = [[self._base + int(bid) * self._block_bytes] for bid in block_ids]
        sizes = [[self._block_bytes] for _ in addrs]
        return addrs, sizes

    def store_routing(self, keys: Sequence[str], block_ids: Sequence[int]) -> None:
        """PUT each block's routing row under ``"re:"+key`` (worker save path).

        Called right after the KV ``batch_put`` for the same request, so the slot
        buffer already holds this request's routing (the forward step scattered
        it). Routing is centralized on ``output_rank`` (the sole slot-buffer
        holder), so — unlike the striped KV PUT — this writes the FULL
        attention-group block set, deduped against the pool via ``batch_is_exist``
        on the ``"re:"`` keys. NO FALLBACK: a negative code raises.
        """
        if not keys:
            return
        self.register()
        re_keys = [_ROUTING_KEY_PREFIX + k for k in keys]
        # Dedup like the KV path: skip rows already in the pool.
        exists = self._store.batch_is_exist(re_keys)
        missing = [i for i, ex in enumerate(exists) if ex != 1]
        if not missing:
            return
        re_keys = [re_keys[i] for i in missing]
        block_ids = [block_ids[i] for i in missing]
        addrs, sizes = self._routing_addrs_sizes(block_ids)
        # Pass the connector's ReplicateConfig (same replica/segment policy as
        # the KV PUT). Omit when None so the store applies its default, matching
        # a connector configured without one.
        if self._replicate_config is not None:
            res = self._store.batch_put_from_multi_buffers(
                re_keys, addrs, sizes, self._replicate_config
            )
        else:
            res = self._store.batch_put_from_multi_buffers(re_keys, addrs, sizes)
        failed = [i for i, v in enumerate(res) if v < 0]
        if failed:
            raise RuntimeError(
                f"Mooncake routing batch_put failed for {len(failed)}/"
                f"{len(re_keys)} keys (codes={set(res[i] for i in failed)})"
            )

    def load_routing(self, keys: Sequence[str], block_ids: Sequence[int]) -> None:
        """GET each block's routing row into the slot buffer (worker load path).

        Called right after the KV ``batch_get`` for the same keys/blocks, so the
        routing lands in the exact slot the scheduler reads for the reloaded
        tokens. NO FALLBACK: a negative code (missing routing for a block whose
        KV was just loaded) raises — a KV-present-but-routing-absent invariant
        violation.
        """
        if not keys:
            return
        self.register()
        re_keys = [_ROUTING_KEY_PREFIX + k for k in keys]
        addrs, sizes = self._routing_addrs_sizes(block_ids)
        res = self._store.batch_get_into_multi_buffers(re_keys, addrs, sizes)
        failed = [i for i, v in enumerate(res) if v < 0]
        if failed:
            raise RuntimeError(
                f"Mooncake routing batch_get failed for {len(failed)}/"
                f"{len(re_keys)} keys (codes={set(res[i] for i in failed)}); "
                "KV was loaded but its routing rows are absent"
            )
