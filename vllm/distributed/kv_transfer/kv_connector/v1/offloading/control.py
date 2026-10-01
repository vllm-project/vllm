# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native CPU operations through the existing offload manager and copy worker."""

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from vllm.distributed.kv_transfer.kv_connector.v1.control import KVTransferQueue
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import CPUControlJob
from vllm.v1.core.kv_cache_transfer import KVCacheTransferLease
from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    get_block_hash,
    get_group_id,
)
from vllm.v1.kv_hints.control import ControlError
from vllm.v1.kv_hints.transfers import KVTransferCompletion
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LookupResult,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        SchedulerOffloadConfig,
    )


@dataclass
class _CPUReservation:
    key: OffloadKey
    context: ReqContext
    is_store: bool
    job: CPUControlJob


class CPUControlAdapter(KVTransferQueue):
    tier: Literal["cpu", "mooncake"] = "cpu"
    actions: tuple[str, ...] = ("demote", "prefetch")
    limitations: tuple[str, ...] = (
        "CPUOffloadingManager with one GPU block per CPU chunk",
        "Full group boundaries and primary cache entries only; PCP=DCP=1",
        "Independent cold retention/deletion and Share are unsupported",
    )

    def __init__(self, manager: CPUOffloadingManager, config: "SchedulerOffloadConfig"):
        super().__init__(config.num_workers)
        self.manager = manager
        self.config = config
        self._groups = {g.group_idx: i for i, g in enumerate(config.kv_group_configs)}
        self._reservations: dict[int, _CPUReservation] = {}

    @property
    def has_pending(self) -> bool:
        return bool(self._reservations)

    def validate(
        self,
        direction: Literal["load", "store"],
        key: BlockHashWithGroupId,
        num_tokens: int,
    ) -> None:
        group = self._groups.get(get_group_id(key))
        if (
            group is None
            or self.config.blocks_per_chunk != 1
            or num_tokens % self.config.kv_group_configs[group].tokens_per_block
        ):
            raise ControlError("unsupported", "Unsupported CPU control boundary/layout")

    def submit(self, lease: KVCacheTransferLease) -> None:
        group = self._groups[get_group_id(lease.key)]
        key = make_offload_key(
            hashlib.sha256(lease.scope.encode() + bytes(lease.key)).digest(),
            get_group_id(lease.key),
        )
        # ReqContext is backend bookkeeping; no inference Request or scheduler
        # queue entry is created for this operation.
        context = ReqContext(f"kv-control:{lease.transfer_id}")
        context.set_offload_key_position(key, lease.num_tokens)
        status = self.manager.lookup(key, context)
        if status in (LookupResult.HIT_PENDING, LookupResult.RETRY):
            raise ControlError("busy", "CPU content is still being prepared")
        if lease.direction == "load" and status == LookupResult.MISS:
            raise ControlError("missing", "CPU content is missing")
        sizes = [0] * len(self._groups)
        indices = [0] * len(self._groups)
        sizes[group] = 1
        indices[group] = (
            lease.num_tokens // self.config.kv_group_configs[group].tokens_per_block - 1
        )
        gpu = GPULoadStoreSpec([lease.block_id], sizes, indices)
        is_store = lease.direction == "store" and status != LookupResult.HIT
        if is_store:
            prepared = self.manager.prepare_store([key], context)
            if prepared is None or prepared.keys_to_store != [key]:
                self.manager.on_request_finished(context)
                raise ControlError("capacity", "CPU store admission was deferred")
            job = CPUControlJob(lease, gpu, prepared.store_spec)
        else:
            source = self.manager.prepare_load([key], context)
            job = (
                CPUControlJob(lease, source, gpu)
                if lease.direction == "load"
                else CPUControlJob(lease, None, None)
            )
        self._reservations[lease.transfer_id] = _CPUReservation(
            key, context, is_store, job
        )
        super().submit(lease)

    def take_jobs(self) -> list[CPUControlJob]:
        return [
            self._reservations[lease.transfer_id].job for lease in self.take_metadata()
        ]

    def poll(self) -> tuple[KVTransferCompletion, ...]:
        completed = super().poll()
        for result in completed:
            reservation = self._reservations.pop(result.transfer_id)
            if reservation.is_store:
                self.manager.complete_store(
                    [reservation.key], reservation.context, success=result.success
                )
            else:
                self.manager.complete_load([reservation.key], reservation.context)
            self.manager.on_request_finished(reservation.context)
        return completed

    def awaiting_load(self, request: Request) -> bool:
        for pending in self._pending.values():
            lease = pending.lease
            if lease.direction != "load":
                continue
            position = lease.num_tokens // self.config.tokens_per_hash - 1
            if 0 <= position < len(request.block_hashes) and request.block_hashes[
                position
            ] == get_block_hash(lease.key):
                return True
        return False
