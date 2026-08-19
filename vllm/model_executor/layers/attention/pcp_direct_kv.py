# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP direct-final KV publication through PyTorch symmetric memory."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.distributed.device_communicators.symm_mem import (
    SymmMemPeerAllocation,
    allocate_symm_mem_peer,
    symm_mem_available,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_MAX_FENCE_SPINS = 100_000_000


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
def _publish_fence_kernel(
    peer_ptrs,
    epoch,
    parity,
    source_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    destination_rank = tl.program_id(0)
    if destination_rank < world_size:
        dest_base = tl.load(peer_ptrs + destination_rank).to(tl.pointer_type(tl.int32))
        tl.atomic_xchg(
            dest_base + parity * world_size + source_rank,
            epoch,
            sem="release",
            scope="sys",
        )


@triton.jit
def _wait_fence_kernel(
    local_signal_ptr,
    epoch,
    parity,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_SPINS: tl.constexpr,
):
    source_rank = tl.arange(0, BLOCK_SIZE)
    mask = source_rank < world_size
    signal_ptr = local_signal_ptr + parity * world_size + source_rank
    observed = tl.atomic_add(signal_ptr, 0, mask=mask, sem="acquire", scope="sys")
    pending = tl.max(tl.where(mask & (observed != epoch), 1, 0))
    spins = 0
    while (pending != 0) & (spins < MAX_SPINS):
        observed = tl.atomic_add(signal_ptr, 0, mask=mask, sem="acquire", scope="sys")
        pending = tl.max(tl.where(mask & (observed != epoch), 1, 0))
        spins += 1
    _trap_if_nonzero(pending)


class PCPPeerCacheFence:
    """Two-kernel release/acquire publication for one PCP group."""

    def __init__(self, group: ProcessGroup, device: torch.device) -> None:
        self._group = group
        self._world_size = group.size()
        self._rank = group.rank()
        self._epoch = 0
        self._allocation = allocate_symm_mem_peer(
            (2, self._world_size),
            dtype=torch.int32,
            device=device,
            group=group,
        )
        self._allocation.storage.zero_()
        torch.accelerator.synchronize()
        dist.barrier(group=group)

    def __call__(self) -> None:
        self._epoch = self._epoch % 0x7FFFFFFE + 1
        parity = self._epoch & 1
        _publish_fence_kernel[(self._world_size,)](
            self._allocation.peer_ptrs,
            self._epoch,
            parity,
            source_rank=self._rank,
            world_size=self._world_size,
        )
        _wait_fence_kernel[(1,)](
            self._allocation.storage,
            self._epoch,
            parity,
            world_size=self._world_size,
            BLOCK_SIZE=triton.next_power_of_2(self._world_size),
            MAX_SPINS=_MAX_FENCE_SPINS,
        )

    def close(self) -> None:
        torch.accelerator.synchronize(self._allocation.storage.device)
        dist.barrier(group=self._group)
        self._allocation.close()


@dataclass
class PCPDirectKVState:
    enabled: bool = False
    world_size: int = 1
    rank: int = 0
    allocations: list[SymmMemPeerAllocation] = field(default_factory=list)
    layer_peer_ptrs: dict[str, torch.Tensor] = field(default_factory=dict)
    layer_mcast_ptrs: dict[str, int] = field(default_factory=dict)
    fence: PCPPeerCacheFence | None = None

    def close(self) -> None:
        if self.fence is not None:
            self.fence.close()
            self.fence = None
        for allocation in self.allocations:
            allocation.close()
        self.allocations.clear()
        self.layer_peer_ptrs.clear()
        self.layer_mcast_ptrs.clear()
        self.enabled = False


_STATE = PCPDirectKVState()


def pcp_direct_kv_requested() -> bool:
    return bool(envs.VLLM_USE_PCP_DIRECT_KV)


def pcp_direct_kv_active() -> bool:
    return _STATE.enabled


def get_pcp_direct_kv_state() -> PCPDirectKVState:
    return _STATE


def get_layer_peer_ptrs(layer_name: str) -> torch.Tensor | None:
    if not _STATE.enabled:
        return None
    return _STATE.layer_peer_ptrs.get(layer_name)


def get_layer_mcast_ptr(layer_name: str) -> int:
    if not _STATE.enabled or not bool(envs.VLLM_PCP_DIRECT_KV_MULTIMEM):
        return 0
    return int(_STATE.layer_mcast_ptrs.get(layer_name, 0))


def publish_pcp_direct_kv() -> None:
    if _STATE.enabled and _STATE.fence is not None:
        _STATE.fence()


def should_allocate_pcp_direct_kv(vllm_config) -> bool:
    if not pcp_direct_kv_requested():
        return False
    if not current_platform.is_cuda() or not symm_mem_available:
        raise RuntimeError(
            "VLLM_USE_PCP_DIRECT_KV=1 requires CUDA torch.distributed._symmetric_memory"
        )
    parallel_config = vllm_config.parallel_config
    if parallel_config.prefill_context_parallel_size <= 1:
        raise RuntimeError(
            "VLLM_USE_PCP_DIRECT_KV=1 requires prefill_context_parallel_size > 1"
        )
    if parallel_config.tensor_parallel_size != 1:
        raise RuntimeError("VLLM_USE_PCP_DIRECT_KV requires tensor_parallel_size=1")
    if parallel_config.decode_context_parallel_size != 1:
        raise RuntimeError(
            "VLLM_USE_PCP_DIRECT_KV requires decode_context_parallel_size=1"
        )
    if parallel_config.data_parallel_size != 1:
        raise RuntimeError("VLLM_USE_PCP_DIRECT_KV requires data_parallel_size=1")
    return True


def allocate_pcp_direct_backing(
    nbytes: int, device: torch.device, group: ProcessGroup
) -> SymmMemPeerAllocation:
    allocation = allocate_symm_mem_peer((nbytes,), torch.int8, device, group)
    _STATE.allocations.append(allocation)
    return allocation


def bind_pcp_direct_layer_views(
    kv_caches: dict[str, object],
    group: ProcessGroup,
    device: torch.device,
) -> None:
    if not _STATE.allocations:
        raise RuntimeError(
            "VLLM_USE_PCP_DIRECT_KV=1 requires every KV buffer to be allocated "
            "with PyTorch symmetric memory"
        )
    layer_peer_ptrs: dict[str, torch.Tensor] = {}
    layer_mcast_ptrs: dict[str, int] = {}
    missing: list[str] = []
    for layer_name, cache in kv_caches.items():
        tensor = _as_cache_tensor(cache)
        if tensor is None:
            continue
        allocation = _allocation_for_tensor(tensor)
        if allocation is None:
            missing.append(layer_name)
            continue
        layer_peer_ptrs[layer_name] = allocation.peer_ptrs_for_view(tensor)
        layer_mcast_ptrs[layer_name] = allocation.multicast_ptr_for_view(tensor)
    if missing:
        raise RuntimeError(
            "VLLM_USE_PCP_DIRECT_KV=1: cache layers not on symmetric-memory "
            f"backing: {', '.join(missing)}"
        )
    if not layer_peer_ptrs:
        raise RuntimeError(
            "VLLM_USE_PCP_DIRECT_KV=1: no bindable KV cache tensors"
        )
    _STATE.layer_peer_ptrs = layer_peer_ptrs
    _STATE.layer_mcast_ptrs = layer_mcast_ptrs
    _STATE.world_size = group.size()
    _STATE.rank = group.rank()
    _STATE.fence = PCPPeerCacheFence(group, device)
    _STATE.enabled = True
    logger.info(
        "PCP direct-KV enabled: world_size=%d layers=%d allocations=%d multicast=%s",
        _STATE.world_size,
        len(layer_peer_ptrs),
        len(_STATE.allocations),
        any(ptr != 0 for ptr in layer_mcast_ptrs.values()),
    )


def close_pcp_direct_kv() -> None:
    _STATE.close()


def _as_cache_tensor(cache: object) -> torch.Tensor | None:
    if isinstance(cache, torch.Tensor):
        return cache
    kv_cache = getattr(cache, "kv_cache", None)
    if isinstance(kv_cache, torch.Tensor):
        return kv_cache
    return None


def _allocation_for_tensor(tensor: torch.Tensor) -> SymmMemPeerAllocation | None:
    storage_ptr = tensor.untyped_storage().data_ptr()
    for allocation in _STATE.allocations:
        if allocation.storage.untyped_storage().data_ptr() == storage_ptr:
            return allocation
    return None


def pcp_group_is_single_node(group) -> bool:
    try:
        return all(in_the_same_node_as(group.cpu_group, source_rank=0))
    except Exception:
        return False
