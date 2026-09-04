# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated direct-KV access through PyTorch symmetric memory."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.device_communicators.symm_mem import (
    SymmetricMemoryAllocation,
    allocate_symmetric_memory,
    rendezvous_symmetric_memory,
)
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)

_MAX_BARRIER_SPINS = 100_000_000


def _default_barrier_index() -> int:
    return 0


@triton.jit
def _trap_if_nonzero(value):
    # Unconditional PTX trap. tl.device_assert is a no-op unless TRITON_DEBUG=1.
    tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.ne.s32 %p0, $1, 0;
            @%p0 trap;
        }
        """,
        "=r, r",
        [value.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _direct_kv_barrier_kernel(
    peer_ptrs,
    offset_bytes,
    local_signal_ptr,
    epoch_ptr,
    barrier_index: tl.constexpr,
    source_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_SPINS: tl.constexpr,
):
    rank = tl.arange(0, BLOCK_SIZE)
    mask = rank < world_size
    epoch = tl.atomic_add(epoch_ptr + barrier_index, 1, sem="relaxed", scope="gpu") + 1
    epoch = epoch.to(tl.uint32)

    parity = epoch & 1
    signal_offset = (barrier_index * 2 + parity) * world_size

    # Publish this rank's completed cache writes to every peer. This kernel is
    # stream-ordered after the cache-write kernel, and the system-scope release
    # makes those preceding writes visible before the epoch is observed.
    ptrs = peer_ptrs.to(tl.uint64).to(tl.pointer_type(tl.uint64))
    dest_base = tl.load(ptrs + rank, mask=mask, other=0).to(tl.pointer_type(tl.uint8))
    dest_signal_ptr = (dest_base + offset_bytes).to(tl.pointer_type(tl.int32))
    tl.atomic_xchg(
        dest_signal_ptr + signal_offset + source_rank,
        epoch,
        mask=mask,
        sem="release",
        scope="sys",
    )
    tl.debug_barrier()

    # Wait until every producer has published the same epoch locally.
    signal_ptr = local_signal_ptr + signal_offset + rank
    observed = tl.atomic_add(signal_ptr, 0, mask=mask, sem="acquire", scope="sys").to(
        tl.uint32
    )
    pending = tl.max(tl.where(mask & (observed != epoch), 1, 0))
    spins = 0
    while (pending != 0) & (spins < MAX_SPINS):
        observed = tl.atomic_add(
            signal_ptr, 0, mask=mask, sem="acquire", scope="sys"
        ).to(tl.uint32)
        pending = tl.max(tl.where(mask & (observed != epoch), 1, 0))
        spins += 1
    _trap_if_nonzero(pending)


def direct_kv_barrier(
    mla_kv_cache: torch.Tensor,
    indexer_k_cache: torch.Tensor | None,
    signal: torch.Tensor,
    epoch: torch.Tensor,
    peer_ptrs: int,
    offset_bytes: int,
    source_rank: int,
    world_size: int,
    barrier_index: int,
) -> None:
    assert 0 <= barrier_index < epoch.numel()
    assert mla_kv_cache.device == signal.device
    assert indexer_k_cache is None or indexer_k_cache.device == signal.device

    _direct_kv_barrier_kernel[(1,)](
        peer_ptrs,
        offset_bytes,
        signal,
        epoch,
        barrier_index,
        source_rank=source_rank,
        world_size=world_size,
        BLOCK_SIZE=triton.next_power_of_2(world_size),
        MAX_SPINS=_MAX_BARRIER_SPINS,
    )


# torch.compile dispatches custom ops on FakeTensors while tracing. The real
# implementation cannot launch a barrier against storage-less tensors or
# advance its epoch during tracing. This op has no outputs, so its fake
# implementation is a no-op; mutates_args below describes its side effects.
def direct_kv_barrier_fake(*_args, **_kwargs) -> None:
    return None


direct_register_custom_op(
    op_name="direct_kv_barrier",
    op_func=direct_kv_barrier,
    # Peer writes mutate the caches; the kernel itself updates signal and epoch.
    # Exposing all four effects preserves write -> barrier -> read ordering.
    mutates_args=["mla_kv_cache", "indexer_k_cache", "signal", "epoch"],
    fake_impl=direct_kv_barrier_fake,
)


def _peer_view_offset_bytes(
    storage: torch.Tensor, handle: Any, view: torch.Tensor
) -> int:
    if view.untyped_storage().data_ptr() != storage.untyped_storage().data_ptr():
        raise ValueError("Cache view does not share its symmetric-memory storage")
    if any(stride < 0 for stride in view.stride()):
        raise ValueError("Negative-stride cache views are unsupported")

    view_offset_bytes = view.data_ptr() - storage.data_ptr()

    span_elements = 0
    if view.numel() != 0:
        span_elements = 1 + sum(
            (size - 1) * stride
            for size, stride in zip(view.shape, view.stride(), strict=True)
        )

    end_bytes = view_offset_bytes + span_elements * view.element_size()
    if view_offset_bytes < 0 or end_bytes > storage.nbytes:
        raise ValueError("Cache view exceeds its symmetric-memory storage")

    return int(handle.offset) + view_offset_bytes


@dataclass(frozen=True)
class KVCacheSymmMemView:
    """Peer pointer table and byte offset for one KV-cache view."""

    peer_ptrs: int
    offset_bytes: int


class KVCacheSymmMemDomain:
    """Group-scoped peer state for symmetric-memory KV-cache storage."""

    def __init__(
        self,
        group: GroupCoordinator,
        num_barriers: int = 1,
        barrier_index_provider: Callable[[], int] = _default_barrier_index,
    ) -> None:
        assert num_barriers >= 1

        self._group = group
        self.world_size = group.world_size
        self._num_barriers = num_barriers
        self._barrier_index_provider = barrier_index_provider
        self._handle: Any | None = None
        self._views: dict[str, KVCacheSymmMemView] = {}
        self._barrier_signals: SymmetricMemoryAllocation | None = None
        self._barrier_epoch: torch.Tensor | None = None

    def rendezvous(self, storage: torch.Tensor) -> None:
        assert self._handle is None
        self._handle = rendezvous_symmetric_memory(storage, self._group.device_group)

    def finalize(
        self,
        kv_caches: Mapping[str, torch.Tensor],
        storage: torch.Tensor,
    ) -> None:
        handle = self._handle
        if handle is None:
            raise RuntimeError(
                "Direct KV requires the KV buffer to be allocated with "
                "PyTorch symmetric memory"
            )

        missing: list[str] = []
        storage_ptr = storage.untyped_storage().data_ptr()

        for layer_name, tensor in kv_caches.items():
            if tensor.untyped_storage().data_ptr() != storage_ptr:
                missing.append(layer_name)
                continue

            self._views[layer_name] = KVCacheSymmMemView(
                peer_ptrs=int(handle.buffer_ptrs_dev),
                offset_bytes=_peer_view_offset_bytes(storage, handle, tensor),
            )

        if missing:
            raise RuntimeError(
                "Direct-KV cache layers not on symmetric-memory storage: "
                f"{', '.join(missing)}"
            )

        if not self._views:
            raise RuntimeError("Direct KV found no bindable KV cache tensors")

        # Device-side epochs advance during every graph replay. Independent
        # barriers allow callers such as DBO to interleave model forwards.
        self._barrier_epoch = torch.zeros(
            self._num_barriers, dtype=torch.int64, device=storage.device
        )
        self._barrier_signals = allocate_symmetric_memory(
            (self._num_barriers, 2, self.world_size),
            dtype=torch.int32,
            device=storage.device,
            group=self._group.device_group,
        )

        logger.info(
            "Replicated direct-KV enabled: world_size=%d views=%d",
            self.world_size,
            len(self._views),
        )

    def view(self, layer_name: str) -> KVCacheSymmMemView:
        try:
            return self._views[layer_name]
        except KeyError as error:
            raise RuntimeError(
                f"Missing direct-KV view for cache layer {layer_name}"
            ) from error

    def barrier(
        self,
        mla_kv_cache: torch.Tensor,
        indexer_k_cache: torch.Tensor | None = None,
    ) -> None:
        signals = self._barrier_signals
        epoch = self._barrier_epoch
        if signals is None or epoch is None:
            raise RuntimeError("Direct-KV cache runtime is not initialized")

        torch.ops.vllm.direct_kv_barrier(
            mla_kv_cache,
            indexer_k_cache,
            signals.storage,
            epoch,
            int(signals.handle.buffer_ptrs_dev),
            int(signals.handle.offset),
            self._group.rank_in_group,
            self.world_size,
            self._barrier_index_provider(),
        )

    def close(self) -> None:
        try:
            if self._barrier_signals is not None:
                torch.accelerator.synchronize(self._barrier_signals.storage.device)
        finally:
            self._barrier_signals = None
            self._barrier_epoch = None
            self._views.clear()
            self._handle = None
