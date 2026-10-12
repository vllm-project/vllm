# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native control adapter using the Store connector's registered buffer layouts."""

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Literal

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.control import (
    KVTransferQueue,
    KVTransferRankResult,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import StoreLayout
from vllm.v1.core.kv_cache_transfer import KVCacheTransferLease
from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    get_block_hash,
    get_group_id,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_hints.control import ControlError
from vllm.v1.request import Request

MOONCAKE_OBJECT_NOT_FOUND = -704


class MooncakeControlAdapter(KVTransferQueue):
    tier: Literal["cpu", "mooncake"] = "mooncake"
    limitations: tuple[str, ...] = (
        "Full group boundaries and primary cache entries only; PCP=DCP=1",
        "Share establishes Store availability; destination GPU requires Prefetch",
        "Control objects use a compatibility namespace separate from automatic offload",
    )

    def __init__(self, config: VllmConfig, cache: KVCacheConfig, hash_block_size: int):
        super().__init__(config.parallel_config.world_size)
        self.parallel = config.parallel_config
        self.cache = cache
        self.cache_hash_block_size = hash_block_size
        assert config.kv_transfer_config is not None
        transfer = config.kv_transfer_config
        can_put = transfer.is_kv_producer or transfer.kv_connector_extra_config.get(
            "save_decode_cache", False
        )
        self.actions: tuple[str, ...] = (
            ("prefetch", "demote", "share") if can_put else ("prefetch",)
        )

    def validate(
        self,
        direction: Literal["load", "store"],
        key: BlockHashWithGroupId,
        num_tokens: int,
    ) -> None:
        group = get_group_id(key)
        if (
            self.parallel.prefill_context_parallel_size != 1
            or self.parallel.decode_context_parallel_size != 1
            or self.cache.hisparse_host_num_blocks is not None
            or group not in self.cache.prefix_cacheable_group_ids
            or num_tokens % self.cache.kv_cache_groups[group].kv_cache_spec.block_size
        ):
            raise ControlError(
                "unsupported", "Unsupported Mooncake control boundary/layout"
            )
        if direction == "store" and "demote" not in self.actions:
            raise ControlError("unsupported", "Connector is not configured to store KV")

    def awaiting_load(self, request: Request) -> bool:
        for pending in self._pending.values():
            lease = pending.lease
            if lease.direction != "load":
                continue
            hash_index = lease.num_tokens // self.cache_hash_block_size - 1
            if 0 <= hash_index < len(request.block_hashes) and request.block_hashes[
                hash_index
            ] == get_block_hash(lease.key):
                return True
        return False


class MooncakeControlWorker:
    def __init__(
        self,
        store: Any,
        layouts: dict[int, StoreLayout],
        *,
        rank: int,
        tp_rank: int,
        group_replication: dict[int, int],
        replicate_config: Any = None,
        group_semantics: bool = False,
    ):
        self.store = store
        self.layouts = layouts
        self.rank = rank
        self.tp_rank = tp_rank
        self.group_replication = group_replication
        self.replicate_config = replicate_config
        self.group_semantics = group_semantics
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kv-control"
        )
        self._pending: dict[int, Future[KVTransferRankResult]] = {}

    def submit(self, leases: list[KVCacheTransferLease], direction: str) -> None:
        selected = [lease for lease in leases if lease.direction == direction]
        if not selected:
            return
        ready = torch.Event()
        ready.record()
        for lease in selected:
            if lease.transfer_id not in self._pending:
                self._pending[lease.transfer_id] = self._executor.submit(
                    self._transfer, lease, ready
                )

    def _transfer(
        self, lease: KVCacheTransferLease, ready: torch.Event
    ) -> KVTransferRankResult:
        try:
            ready.synchronize()
            group = get_group_id(lease.key)
            layout = self.layouts[group]
            if lease.num_tokens % layout.block_size:
                raise ValueError("Control transfer splits a Store block")
            factor = self.group_replication[group]
            block_index = lease.num_tokens // layout.block_size - 1
            if lease.direction == "store" and (
                block_index % factor
                != (self.tp_rank + layout.metadata.group_id) % factor
            ):
                return KVTransferRankResult(
                    lease.transfer_id, lease.generation, self.rank, True
                )
            shards = layout.local_shard_ids
            block_hash = get_block_hash(lease.key)
            keys = [
                f"kv-control:{lease.scope}@{layout.key_for(shard, block_hash)}"
                for shard in shards
            ]
            addresses, sizes, _ = layout.prepare_values(
                [(0, layout.block_size)] * len(shards), [lease.block_id], shards
            )
            if not addresses or any(not size for size in sizes):
                raise ValueError("No registered Store buffers for this group")
            if lease.direction == "load":
                result = self.store.batch_get_into_multi_buffers(keys, addresses, sizes)
            else:
                if self.group_semantics:
                    self.replicate_config.group_ids = [
                        f"kv-control:{lease.scope}@{get_block_hash(lease.key).hex()}"
                    ] * len(keys)
                if self.replicate_config is None:
                    result = self.store.batch_put_from_multi_buffers(
                        keys, addresses, sizes
                    )
                else:
                    result = self.store.batch_put_from_multi_buffers(
                        keys, addresses, sizes, self.replicate_config
                    )
            failures = [value for value in result if value < 0]
            if (
                lease.direction == "load"
                and len(result) == len(keys)
                and failures
                and all(value == MOONCAKE_OBJECT_NOT_FOUND for value in failures)
            ):
                return KVTransferRankResult(
                    lease.transfer_id,
                    lease.generation,
                    self.rank,
                    False,
                    "Mooncake object is missing",
                    missing=True,
                )
            if len(result) != len(keys) or failures:
                raise RuntimeError(f"Mooncake {lease.direction} failed: {result}")
            if lease.direction == "load" and any(
                value != sum(size) for value, size in zip(result, sizes, strict=True)
            ):
                raise RuntimeError("Mooncake load returned an incomplete object")
            return KVTransferRankResult(
                lease.transfer_id,
                lease.generation,
                self.rank,
                True,
                bytes_transferred=sum(sum(size) for size in sizes),
            )
        except Exception as error:
            return KVTransferRankResult(
                lease.transfer_id,
                lease.generation,
                self.rank,
                False,
                f"{type(error).__name__}: {error}"[:512],
            )

    def poll(self) -> list[KVTransferRankResult]:
        completed = []
        for transfer_id, future in tuple(self._pending.items()):
            if future.done():
                completed.append(future.result())
                del self._pending[transfer_id]
        return completed

    def close(self) -> None:
        self._executor.shutdown(wait=True)
