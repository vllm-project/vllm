# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct-final stores/barrier over ExtensibleKVCache-owned CUDA VMM storage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.extensible_tensor import ExtensibleTensor
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

_MAX_BARRIER_SPINS = 100_000_000


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


@dataclass(frozen=True)
class KVCacheVmmView:
    peer_ptrs: int
    offset_bytes: int


class KVCacheVmmDomain:
    """Peer views borrow the cache owner's PA; no cache backing is allocated here."""

    def __init__(self, group: GroupCoordinator, owner: ExtensibleTensor):
        self.group = group
        self.world_size = group.world_size
        self.owner = owner
        self.peers = owner.share_with(group.cpu_group)
        self._views: dict[str, KVCacheVmmView] = {}
        self._layouts: tuple = ()
        error = None
        try:
            self.signals = ExtensibleTensor(
                2 * self.world_size * 4, device=owner.device, exportable=True
            )
            self.signals.resize_per_segment_(2 * self.world_size * 4, zero_new=True)
            self.epoch = torch.zeros(1, dtype=torch.int64, device=owner.device)
        except RuntimeError as exc:
            error = exc
        self.peers._check(error)
        self.signal_peers = self.signals.share_with(group.cpu_group)
        self.signal_view = self.signals.full_view().view(torch.int32)
        self._closed = False

    def bind(self, caches: Mapping[str, torch.Tensor], layers: Mapping[str, Any]):
        error = None
        try:
            layouts, views = self._validate_views(caches, layers)
        except RuntimeError as exc:
            error = exc
        self.peers._check(error)
        self.peers._identical(layouts)
        # The block count may change after capture; strides and addresses may not.
        stable_layouts = tuple((n, s[1:], st, dt, off) for n, s, st, dt, off in layouts)
        self.peers._check(
            RuntimeError("KV peer layout changed after graph capture")
            if self._layouts and self._layouts != stable_layouts
            else None
        )
        self._layouts = stable_layouts
        self._views = views
        for layer in layers.values():
            if getattr(layer, "use_pcp", False):
                layer.pcp_vmm_domain = self

    def _validate_views(self, caches, layers):
        layouts = []
        views = {}
        for name, tensor in sorted(caches.items()):
            offset = tensor.data_ptr() - self.owner.base_ptr
            span = (
                0
                if not tensor.numel()
                else (
                    1
                    + sum(
                        (n - 1) * s
                        for n, s in zip(tensor.shape, tensor.stride(), strict=True)
                    )
                )
                * tensor.element_size()
            )
            if offset < 0 or offset + span > self.owner._max_num_bytes:
                raise RuntimeError(f"KV view {name} is not owned by ExtensibleKVCache")
            layouts.append(
                (
                    name,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    str(tensor.dtype),
                    offset,
                )
            )
            view = KVCacheVmmView(self.peers.pointers.data_ptr(), offset)
            if name in self._views and view != self._views[name]:
                raise RuntimeError("KV peer offsets changed after graph capture")
            views[name] = view
        for layer in layers.values():
            if getattr(layer, "use_pcp", False) and not hasattr(
                layer, "pcp_vmm_domain"
            ):
                raise RuntimeError(
                    "A PCP attention layer does not support direct-final KV"
                )
        return tuple(layouts), views

    def view(self, name: str) -> KVCacheVmmView:
        if self._closed or self.peers._failed:
            raise RuntimeError("Direct-final VMM domain is unavailable")
        return self._views[name]

    def barrier(
        self, mla_kv_cache: torch.Tensor, indexer_k_cache: torch.Tensor | None = None
    ):
        torch.ops.vllm.direct_kv_barrier(
            mla_kv_cache,
            indexer_k_cache,
            self.signal_view,
            self.epoch,
            self.signal_peers.pointers.data_ptr(),
            0,
            self.group.rank_in_group,
            self.world_size,
            0,
        )

    def close(self):
        if self._closed:
            return
        self.signals.free()
        self.peers.close()
        self._views.clear()
        self._closed = True


def validate_direct_final(config, extensible: bool) -> None:
    """Initial supported intersection; requested direct mode never falls back."""
    parallel = config.parallel_config
    reasons = []
    from vllm.platforms import current_platform

    if not current_platform.is_cuda() or not extensible:
        reasons.append("CUDA ExtensibleKVCache backing is required")
    if (
        parallel.prefill_context_parallel_size <= 1
        or parallel.decode_context_parallel_size != 1
    ):
        reasons.append("replicated PCP>1 and DCP1 are required")
    if parallel.pipeline_parallel_size != 1 or parallel.num_ubatches > 1:
        reasons.append("pipeline parallelism and microbatch overlap are not supported")
    if config.speculative_config is not None or config.kv_transfer_config is not None:
        reasons.append(
            "speculation and KV transfer are not supported in this first version"
        )
    if config.model_config.enable_sleep_mode:
        reasons.append("sleep mode is not enabled in this first version")
    if config.attention_config.hisparse_config is not None:
        reasons.append("HiSparse is not supported")
    if config.attention_config.resolve_indexer_kv_dtype("fp8") != "fp8":
        reasons.append("the existing direct-final Indexer writer requires FP8")
    if config.cache_config.cache_dtype not in (
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ):
        reasons.append("unsupported KV dtype for the existing direct-final writer")
    layers = config.compilation_config.static_forward_context
    pcp_layers = [
        layer for layer in layers.values() if getattr(layer, "use_pcp", False)
    ]
    if not pcp_layers or any(
        not hasattr(layer, "pcp_vmm_domain") for layer in pcp_layers
    ):
        reasons.append("all PCP layers must support direct-final publication")
    if reasons:
        raise ValueError("VLLM_USE_PCP_DIRECT_KV=1: " + "; ".join(reasons))
