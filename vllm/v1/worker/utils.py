# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product as iprod
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from vllm.config import CacheConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mamba.mamba_mixer2 import share_replayssm_ring_trackers
from vllm.model_executor.layers.utils import warmup_rocm_skinny_gemm_workspaces
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
    create_kv_cache_views,
)
from vllm.v1.worker.block_table import get_block_table_width

logger = init_logger(__name__)


@runtime_checkable
class KVCacheSideStorageLayer(Protocol):
    """Lifecycle contract for storage aligned with an owner KV cache."""

    main_layer_name: str

    def bind_kv_cache_side_storage(self, forward_context: dict[str, Any]) -> None: ...

    def zero_kv_cache_side_storage(
        self, block_ids: torch.Tensor | list[int]
    ) -> None: ...

    def reset_kv_cache_side_storage_runtime_state(
        self,
    ) -> tuple[int, int]: ...

    def copy_kv_cache_side_storage(
        self,
        block_copies: Sequence[KVCacheBlockCopy],
        num_blocks: int,
    ) -> None: ...


@runtime_checkable
class KVCacheSideStorageTransfer(Protocol):
    """Side storage that KV connectors move together with its owner's blocks."""

    main_layer_name: str

    def kv_cache_side_storage_transfer_view(self) -> torch.Tensor:
        """Return a contiguous ``[num_blocks, bytes]`` view of full blocks."""
        ...


@runtime_checkable
class KVCacheSideStorageRequestLifecycle(Protocol):
    """Optional request-lifetime state associated with physical KV blocks."""

    main_layer_name: str

    def release_kv_cache_side_storage(
        self, block_ids: torch.Tensor | list[int]
    ) -> None: ...


@runtime_checkable
class KVCacheSideStorageMemoryProfiler(Protocol):
    """Optional fixed-memory profiling contract for KV side storage.

    ``memory_profile_key`` is compared by object identity. Layers that share
    one fixed workspace should return the same owner object so the generic
    worker lifecycle invokes their preparation hook exactly once.
    """

    def kv_cache_side_storage_memory_profile_key(
        self,
        forward_context: dict[str, Any],
    ) -> object: ...

    def prepare_kv_cache_side_storage_memory_profile(
        self,
        forward_context: dict[str, Any],
        device: torch.device,
    ) -> None: ...


def _iter_unique_layers(
    forward_context: dict[str, Any],
    layer_type: type,
) -> Iterator[tuple[str, Any]]:
    """Yield the first ``(name, layer)`` for each distinct ``layer_type`` layer."""
    seen_layers: set[int] = set()
    for layer_name, layer in forward_context.items():
        layer_id = id(layer)
        if layer_id in seen_layers:
            continue
        seen_layers.add(layer_id)
        if isinstance(layer, layer_type):
            yield layer_name, layer


def _layer_name_to_group_id(
    kv_cache_groups: Sequence[KVCacheGroupSpec],
) -> dict[str, int]:
    return {
        layer_name: group_id
        for group_id, group in enumerate(kv_cache_groups)
        for layer_name in group.layer_names
    }


def get_kv_cache_side_storage_transfer_views(
    static_forward_context: dict[str, Any],
    kv_cache_groups: Sequence[KVCacheGroupSpec],
) -> dict[int, list[torch.Tensor]]:
    """Group the transferable side-storage views by their owner KV cache group."""
    group_ids = _layer_name_to_group_id(kv_cache_groups)
    views: dict[int, list[torch.Tensor]] = {}
    for _, layer in _iter_unique_layers(
        static_forward_context, KVCacheSideStorageTransfer
    ):
        views.setdefault(group_ids[layer.main_layer_name], []).append(
            layer.kv_cache_side_storage_transfer_view()
        )
    return views


def prepare_kv_cache_side_storage_for_memory_profiling(
    static_forward_context: dict[str, Any],
    device: torch.device,
) -> int:
    """Allocate and retain fixed side-storage memory before model profiling.

    This must run inside the worker's main ``memory_profiling`` interval.
    Persistent allocations made here are then included in the non-KV memory
    baseline before the KV block budget is computed.

    Returns the number of distinct profiling owners that were prepared.
    """
    seen_profile_owners: set[int] = set()
    num_profile_owners = 0
    for _, layer in _iter_unique_layers(
        static_forward_context, KVCacheSideStorageMemoryProfiler
    ):
        profile_owner = layer.kv_cache_side_storage_memory_profile_key(
            static_forward_context
        )
        profile_owner_id = id(profile_owner)
        if profile_owner_id in seen_profile_owners:
            continue
        seen_profile_owners.add(profile_owner_id)
        layer.prepare_kv_cache_side_storage_memory_profile(
            static_forward_context,
            device,
        )
        num_profile_owners += 1
    return num_profile_owners


def reset_kv_cache_side_storage_runtime_state(
    static_forward_context: dict[str, Any],
) -> tuple[int, int]:
    """Reset model-specific side storage after startup dummy executions.

    Hooks must update existing tensors in place because CUDA graphs retain
    their addresses.
    """
    num_storages = 0
    num_scratch_buffers = 0
    for _, layer in _iter_unique_layers(
        static_forward_context, KVCacheSideStorageLayer
    ):
        storage_count, scratch_buffer_count = (
            layer.reset_kv_cache_side_storage_runtime_state()
        )
        num_storages += int(storage_count)
        num_scratch_buffers += int(scratch_buffer_count)
    return num_storages, num_scratch_buffers


def reset_kv_cache_side_storage_and_log(
    static_forward_context: dict[str, Any],
    phase: str,
) -> None:
    num_side_storages, num_scratch_buffers = reset_kv_cache_side_storage_runtime_state(
        static_forward_context
    )
    if num_side_storages:
        torch.accelerator.synchronize()
        logger.info(
            "Reset KV side-storage state after %s: storages=%d scratch_buffers=%d",
            phase,
            num_side_storages,
            num_scratch_buffers,
        )


def raise_if_nan_logits(num_nans_in_logits: Mapping[str, int]) -> None:
    if not any(num_nans_in_logits.values()):
        return

    corrupted_requests = {
        req_id: num_nans
        for req_id, num_nans in num_nans_in_logits.items()
        if num_nans > 0
    }
    raise RuntimeError(f"NaNs detected in logits: {corrupted_requests}")


@triton.jit
def _zero_kv_blocks_kernel(
    seg_addrs_ptr,
    seg_block_strides_ptr,
    seg_page_sizes_ptr,
    block_ids_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Zero KV cache blocks across all segments in a single launch.

    Each segment is a contiguous region of one block's data.  Layer-compact
    layouts have one segment per layer buffer; dimensions physically outside
    the block dim (separate head groups under LHBNC) and virtual block splits
    each get their own segment.

    Segments may have different block strides and page sizes (e.g. packed
    KV views or models with multiple KV cache groups like MLA + DSA
    indexer). Each segment's block stride determines where a logical block
    begins, while its page size determines how many elements are cleared.

    seg_addrs_ptr holds absolute byte addresses (int64) for each segment,
    allowing segments to live in different CUDA allocations.

    Programs are mapped directly onto a 3-D grid as
    (block_index, seg_index, chunk_index).
    """
    block_index = tl.program_id(0)
    seg_index = tl.program_id(1)
    chunk_index = tl.program_id(2)
    block_stride_el = tl.load(seg_block_strides_ptr + seg_index)
    page_size_el = tl.load(seg_page_sizes_ptr + seg_index)
    chunk_offset = chunk_index.to(tl.int64) * BLOCK_SIZE
    if chunk_offset >= page_size_el:
        return
    block_id = tl.load(block_ids_ptr + block_index)
    seg_addr = tl.load(seg_addrs_ptr + seg_index)
    ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
    block_offset = block_id.to(tl.int64) * block_stride_el.to(tl.int64)
    cols = chunk_offset + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    tl.store(
        ptr + block_offset + cols,
        tl.zeros([BLOCK_SIZE], dtype=tl.int32),
        mask=cols < page_size_el,
    )


class KVBlockZeroer:
    """Manages efficient zeroing of KV cache blocks via a Triton kernel.

    Construct once after KV caches are allocated to precompute segment
    addresses, then call :meth:`zero_block_ids` each step to zero
    newly-allocated blocks.
    """

    def __init__(
        self,
        device: torch.device,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[int],
        static_forward_context: dict[str, Any],
        num_blocks: int,
        runner_only_attn_layers: set[str] | None = None,
        zero_base_kv_cache: bool = True,
    ) -> None:
        """Precompute the absolute-address table for the Triton zeroing kernel.

        Each entry is the absolute byte address of a segment start on the
        GPU, so segments in different CUDA allocations work correctly.

        Per-layer views are standardized ``[B, H, N, C]`` with blocks at dim 0; dims
        physically outside B (separate head groups under LHBNC) each get their own
        segment. A segment's page spans everything inside its block, so under BHLNC it
        also covers the block's other layers -- safe, since block IDs are global pool
        indices and a newly allocated block owns its whole tile.

        Block IDs from the scheduler reference logical blocks whose size
        may differ from the kernel block size (virtual block splitting).
        Each virtual block is represented as an independent segment so its
        physical block stride and zeroed page span remain independent.

        Only AttentionSpec layers are processed; Mamba layers are skipped.
        ``zero_base_kv_cache=False`` keeps only side-storage lifecycle hooks,
        avoiding full KV-page writes when the sidecar alone requires reset.
        """
        self.device = device
        self._meta: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]
            | tuple[torch.Tensor, torch.Tensor, int, int, int]
            | None
        ) = None
        self._side_storage_zero_hooks: dict[int, list[Any]] = {}

        if runner_only_attn_layers is None:
            runner_only_attn_layers = set()
        # Packed layouts and cross-layer sharing can expose the same address
        # through multiple logical layers.  Keep the widest page span for an
        # overlaid address, but reject conflicting logical block strides since
        # one segment cannot represent two different block-index geometries.
        seen_ptrs: dict[int, int] = {}
        seg_addrs: list[int] = []
        seg_block_strides: list[int] = []
        seg_page_sizes: list[int] = []

        attn_groups = list(attn_groups_iter)
        layer_name_to_group_id: dict[str, int] = {}
        for group in attn_groups:
            for layer_name in group.layer_names:
                previous_group_id = layer_name_to_group_id.setdefault(
                    layer_name, group.kv_cache_group_id
                )
                if previous_group_id != group.kv_cache_group_id:
                    raise ValueError(
                        "KV cache layer belongs to multiple groups: "
                        f"layer={layer_name!r}, groups="
                        f"{previous_group_id},{group.kv_cache_group_id}"
                    )

        for layer_name, layer in _iter_unique_layers(
            static_forward_context, KVCacheSideStorageLayer
        ):
            owner_name = layer.main_layer_name
            group_id = layer_name_to_group_id.get(owner_name)
            if group_id is None:
                raise ValueError(
                    "KV side-storage layer has no owner KV cache group: "
                    f"layer={layer_name!r}, owner={owner_name!r}"
                )
            self._side_storage_zero_hooks.setdefault(group_id, []).append(
                layer.zero_kv_cache_side_storage
            )
        for group in attn_groups:
            if (
                group.kv_cache_spec.requires_zeroing
                and group.kv_cache_group_id not in self._side_storage_zero_hooks
            ):
                raise ValueError(
                    "KV cache group declares side-storage zeroing but no "
                    "KVCacheSideStorageLayer owns the group: "
                    f"group={group.kv_cache_group_id}, "
                    f"layers={group.layer_names!r}"
                )

        if zero_base_kv_cache:
            for group in attn_groups:
                spec = group.kv_cache_spec
                if not isinstance(spec, AttentionSpec):
                    continue
                if group.kv_cache_group_id >= len(kernel_block_sizes):
                    continue
                kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
                if kernel_bs <= 0 or spec.block_size % kernel_bs != 0:
                    raise ValueError(
                        "KV cache block size must be divisible by the kernel "
                        f"block size: spec={spec.block_size}, kernel={kernel_bs}"
                    )
                block_dim = 0

                for layer_name in group.layer_names:
                    if layer_name in runner_only_attn_layers:
                        continue
                    kv = static_forward_context[layer_name].kv_cache
                    if not isinstance(kv, torch.Tensor):
                        continue
                    if group.kernel_block_stride:
                        kv = group.map_kv_cache_to_manager_blocks(kv, num_blocks)
                    if kv.device.type != self.device.type:
                        continue
                    ratio, remainder = divmod(kv.shape[0], num_blocks)
                    if remainder or ratio <= 0:
                        raise ValueError(
                            "KV kernel blocks must divide into logical blocks"
                        )
                    el = kv.element_size()
                    block_stride_bytes = kv.stride(block_dim) * el
                    if block_stride_bytes <= 0 or block_stride_bytes % 4 != 0:
                        raise ValueError(
                            "KV block stride must be a positive multiple of "
                            f"4 bytes, got {block_stride_bytes}"
                        )
                    if kv.shape[block_dim] % ratio != 0:
                        raise ValueError(
                            "KV cache physical block count must be divisible by "
                            f"virtual split ratio: shape={tuple(kv.shape)}, "
                            f"block_dim={block_dim}, ratio={ratio}"
                        )

                    # Find the largest contiguous run inside one physical
                    # block and split every non-dense dimension into its own
                    # segment.  Merely treating dimensions whose stride is
                    # larger than the block stride as "outer" is insufficient
                    # for strided/padded views: it can make the kernel write
                    # across a gap into a neighbouring layer or page.
                    #
                    # Sort by physical stride rather than logical dimension
                    # order so this works for both blocks-first and K/V-first
                    # layouts.  Once a gap is encountered, all larger-stride
                    # dimensions are independent planes; only dimensions of
                    # size one can be ignored safely.
                    dims_by_stride = sorted(
                        (
                            d
                            for d in range(kv.ndim)
                            if d != block_dim and kv.shape[d] > 1
                        ),
                        key=lambda d: (kv.stride(d), d),
                    )
                    kernel_page_bytes = el
                    split_dims: list[int] = []
                    dense = True
                    for dim in dims_by_stride:
                        dim_stride_bytes = kv.stride(dim) * el
                        if dense and dim_stride_bytes == kernel_page_bytes:
                            kernel_page_bytes *= kv.shape[dim]
                        else:
                            dense = False
                            split_dims.append(dim)
                    if kernel_page_bytes <= 0 or kernel_page_bytes % 4 != 0:
                        raise ValueError(
                            "KV page span must be a positive multiple of 4 "
                            f"bytes, got {kernel_page_bytes}"
                        )
                    if kernel_page_bytes > block_stride_bytes:
                        raise ValueError(
                            "KV page span exceeds physical block stride; "
                            f"page={kernel_page_bytes}, stride={block_stride_bytes}, "
                            f"shape={tuple(kv.shape)}, strides={tuple(kv.stride())}"
                        )
                    logical_block_stride_bytes = block_stride_bytes * ratio
                    split_ranges = (range(kv.shape[d]) for d in split_dims)
                    for split_indices in iprod(*split_ranges):
                        off_bytes = sum(
                            i * kv.stride(d) * el
                            for i, d in zip(split_indices, split_dims)
                        )
                        for virtual_index in range(ratio):
                            addr = (
                                kv.data_ptr()
                                + off_bytes
                                + virtual_index * block_stride_bytes
                            )
                            if addr % 4 != 0:
                                raise ValueError(
                                    f"KV segment address is not 4-byte aligned: {addr}"
                                )
                            page_size_el = kernel_page_bytes // 4
                            block_stride_el = logical_block_stride_bytes // 4
                            existing = seen_ptrs.get(addr)
                            if existing is not None:
                                if seg_block_strides[existing] != block_stride_el:
                                    raise ValueError(
                                        "Overlaid KV segments have conflicting "
                                        "logical block strides: "
                                        f"{seg_block_strides[existing]} vs "
                                        f"{block_stride_el}"
                                    )
                                seg_page_sizes[existing] = max(
                                    seg_page_sizes[existing], page_size_el
                                )
                                continue
                            seen_ptrs[addr] = len(seg_addrs)
                            seg_addrs.append(addr)
                            seg_block_strides.append(block_stride_el)
                            seg_page_sizes.append(page_size_el)

        if not seg_addrs:
            self._meta = None
            return

        max_page_size_el = max(seg_page_sizes)
        # A fixed power-of-two tile keeps the launch bounded even when one
        # layer has a very large page and another has a tiny one.  Individual
        # pages may not be exact multiples of this tile, so the kernel masks
        # the final partial chunk.
        blk_size = min(1 << (max_page_size_el - 1).bit_length(), 1024)
        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
            torch.tensor(seg_block_strides, dtype=torch.int64, device=self.device),
            torch.tensor(seg_page_sizes, dtype=torch.int64, device=self.device),
            (max_page_size_el + blk_size - 1) // blk_size,
            blk_size,
            len(seg_addrs),
        )

    def zero_block_ids(
        self,
        block_ids: list[int],
        block_ids_by_group: list[list[int]] | None = None,
    ) -> None:
        """Zero base KV blocks and side storage owned by each KV group."""
        if block_ids and self._meta is not None:
            # Accept the pre-geometry five-field metadata shape used by older
            # callers/tests.  In that format each page was assumed to be
            # contiguous and therefore also served as the block stride.
            if len(self._meta) == 5:
                (
                    seg_addrs,
                    seg_page_sizes,
                    max_chunks,
                    blk_size,
                    n_segs,
                ) = self._meta
                seg_block_strides = seg_page_sizes
            else:
                (
                    seg_addrs,
                    seg_block_strides,
                    seg_page_sizes,
                    max_chunks,
                    blk_size,
                    n_segs,
                ) = self._meta
            n_blocks = len(block_ids)
            idx = async_tensor_h2d(block_ids, device=self.device, dtype=torch.int64)
            grid = (n_blocks, n_segs, max_chunks)
            _zero_kv_blocks_kernel[grid](
                seg_addrs,
                seg_block_strides,
                seg_page_sizes,
                idx,
                BLOCK_SIZE=blk_size,
            )

        side_storage_zero_hooks = getattr(
            self,
            "_side_storage_zero_hooks",
            {},
        )
        if not side_storage_zero_hooks:
            return
        if block_ids_by_group is None:
            if block_ids:
                raise RuntimeError(
                    "KV side-storage reset requires block IDs grouped by KV "
                    "cache group."
                )
            return
        required_group_count = max(side_storage_zero_hooks) + 1
        if len(block_ids_by_group) < required_group_count:
            raise RuntimeError(
                "KV side-storage reset is missing owner group IDs: "
                f"required_groups={required_group_count}, "
                f"actual_groups={len(block_ids_by_group)}"
            )

        flat_group_ids = [
            block_id for group_ids in block_ids_by_group for block_id in group_ids
        ]
        if not flat_group_ids:
            return
        group_ids_gpu = async_tensor_h2d(
            flat_group_ids,
            device=self.device,
            dtype=torch.int64,
        )
        offset = 0
        for group_id, group_ids in enumerate(block_ids_by_group):
            next_offset = offset + len(group_ids)
            hooks = side_storage_zero_hooks.get(group_id, ())
            if hooks and next_offset > offset:
                ids = group_ids_gpu[offset:next_offset]
                for hook in hooks:
                    hook(ids)
            offset = next_offset

    def warmup(self, num_kv_blocks: int) -> None:
        """JIT-compile the zeroing kernel before the first real request."""
        if num_kv_blocks > 0:
            side_storage_zero_hooks = getattr(
                self,
                "_side_storage_zero_hooks",
                {},
            )
            num_groups = max(side_storage_zero_hooks, default=-1) + 1
            block_ids_by_group = [
                [0] if group_id in side_storage_zero_hooks else []
                for group_id in range(num_groups)
            ]
            self.zero_block_ids([0], block_ids_by_group)


@dataclass(eq=False)
class AttentionGroup:
    backend: type[AttentionBackend]
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    kv_cache_group_id: int
    # Set when the backend needs blocks smaller than the KV cache manager's: its
    # block tables and KV cache views then address kernel blocks, kernel block j
    # of block b being b * kernel_block_stride + j. The stride is set when a
    # packed KV cache is bound (other layers' pages sit between its blocks);
    # otherwise it stays 0 and the kernel blocks per block are used.
    kernel_block_size: int | None = None
    kernel_block_stride: int = 0
    # Persistent per ubatch: CUDA graphs capture the mapped block table.
    kernel_block_table: torch.Tensor | None = None
    kernel_block_offsets: torch.Tensor | None = None
    # When ubatching is enabled we will have a metadata builder for each ubatch
    # so that if they use internal persistent buffers for cudagraphs, and they
    # won't have to worry about conflicting with the other ubatches.
    metadata_builders: list[AttentionMetadataBuilder] = field(
        default_factory=lambda: []
    )

    def create_metadata_builders(
        self,
        vllm_config,
        device,
        kernel_block_size: int | None = None,
        num_metadata_builders: int = 1,
    ):
        spec = self.kv_cache_spec
        if kernel_block_size is not None and isinstance(spec, AttentionSpec):
            # A backend in a packed KV cache group may need smaller blocks than
            # the group is split into.
            kernel_block_size = select_common_block_size(
                kernel_block_size, [self.backend], [spec]
            )
            if kernel_block_size != spec.block_size:
                self.kernel_block_size = kernel_block_size
        if kernel_block_size is None:
            kv_cache_spec_builder = self.kv_cache_spec
        else:
            kv_cache_spec_builder = self.kv_cache_spec.copy_with_new_block_size(
                kernel_block_size
            )
        builder_cls = self.backend.get_builder_cls()
        builder_kwargs = {}
        if builder_cls.requires_block_table_width or self.kernel_block_size is not None:
            max_num_blocks = self.kv_cache_spec.max_num_blocks_per_req(
                vllm_config, vllm_config.model_config.max_model_len
            )
            width = get_block_table_width(max_num_blocks, self.kv_cache_spec.block_size)
            if kernel_block_size is not None:
                width *= self.kv_cache_spec.block_size // kernel_block_size
            if builder_cls.requires_block_table_width:
                builder_kwargs["block_table_width"] = width
        if self.kernel_block_size is not None:
            self.kernel_block_table = torch.zeros(
                num_metadata_builders,
                vllm_config.scheduler_config.max_num_seqs,
                width,
                dtype=torch.int32,
                device=device,
            )
            self.kernel_block_offsets = torch.arange(
                spec.block_size // self.kernel_block_size,
                dtype=torch.int32,
                device=device,
            )
        self.metadata_builders = [
            builder_cls(
                kv_cache_spec_builder,
                self.layer_names,
                vllm_config,
                device,
                **builder_kwargs,
            )
            for _ in range(num_metadata_builders)
        ]
        for ubatch_id, builder in enumerate(self.metadata_builders):
            builder.ubatch_id = ubatch_id

    def build_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        ubatch_idx: int = 0,
        common_prefix_len: int = 0,
        **kwargs,
    ) -> Any:
        return self.get_metadata_builder(ubatch_idx).build(
            common_prefix_len,
            self._with_kernel_block_table(common_attn_metadata, ubatch_idx),
            **kwargs,
        )

    def build_metadata_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata, ubatch_idx: int = 0
    ) -> Any:
        return self.get_metadata_builder(ubatch_idx).build_for_cudagraph_capture(
            self._with_kernel_block_table(common_attn_metadata, ubatch_idx)
        )

    def build_metadata_for_drafting(
        self, common_attn_metadata: CommonAttentionMetadata, draft_index: int
    ) -> Any:
        return self.get_metadata_builder().build_for_drafting(
            self._with_kernel_block_table(common_attn_metadata), draft_index
        )

    def _with_kernel_block_table(
        self, common_attn_metadata: CommonAttentionMetadata, ubatch_idx: int = 0
    ) -> CommonAttentionMetadata:
        if self.kernel_block_size is None:
            return common_attn_metadata
        return replace(
            common_attn_metadata,
            block_table_tensor=self.map_to_kernel_block_table(
                common_attn_metadata.block_table_tensor, ubatch_idx
            ),
        )

    def map_to_kernel_block_table(
        self, block_table: torch.Tensor, ubatch_idx: int = 0
    ) -> torch.Tensor:
        if self.kernel_block_size is None:
            return block_table
        assert self.kernel_block_table is not None
        blocks_per_kv_block = self.kv_cache_spec.block_size // self.kernel_block_size
        rows, cols = block_table.shape
        out = self.kernel_block_table[ubatch_idx, :rows, : cols * blocks_per_kv_block]
        torch.add(
            self.kernel_block_offsets,
            block_table.unsqueeze(-1),
            alpha=self.kernel_block_stride or blocks_per_kv_block,
            out=out.unflatten(1, (cols, -1)),
        )
        return out

    def map_kv_cache_to_kernel_blocks(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """Views a packed cache in manager blocks as kernel blocks."""
        assert self.kernel_block_size is not None
        spec = self.kv_cache_spec
        kernel_rows = spec.get_num_kernel_states(self.kernel_block_size)
        if kv_cache.shape[-2] == kernel_rows:
            return kv_cache
        # Token and pooled caches are written through manager-block slot
        # mappings, so only compressed caches (written from their block table)
        # can be mapped.
        if (
            spec.tokens_per_state <= 1
            or kv_cache.shape[1] != 1
            or kv_cache.stride(0) % (kernel_rows * kv_cache.stride(-2))
        ):
            raise ValueError(
                f"{self.backend.get_name()} needs {self.kernel_block_size}-token "
                f"kernel blocks, which packed {spec.block_size}-token KV cache "
                f"blocks cannot be split into. Reduce --block-size to "
                f"{self.kernel_block_size} or set VLLM_KV_CACHE_LAYOUT to a "
                "layer-compact layout (e.g. LBNHC)."
            )
        # Kernel block j of block b is b * kernel_block_stride + j. The view
        # spans the other layers' pages between blocks (never addressing them),
        # which ``view`` cannot express.
        page_stride = kernel_rows * kv_cache.stride(-2)
        self.kernel_block_stride = kv_cache.stride(0) // page_stride
        num_pages = (kv_cache.shape[0] - 1) * self.kernel_block_stride + (
            kv_cache.shape[-2] // kernel_rows
        )
        return kv_cache.as_strided(
            (num_pages, *kv_cache.shape[1:-2], kernel_rows, kv_cache.shape[-1]),
            (page_stride, *kv_cache.stride()[1:]),
        )

    def map_kv_cache_to_manager_blocks(
        self, kv_cache: torch.Tensor, num_blocks: int
    ) -> torch.Tensor:
        """Views a packed cache in kernel blocks as manager blocks."""
        assert self.kernel_block_stride
        spec = self.kv_cache_spec
        rows = spec.get_num_kernel_states(spec.block_size)
        return kv_cache.as_strided(
            (num_blocks, *kv_cache.shape[1:-2], rows, kv_cache.shape[-1]),
            (kv_cache.stride(0) * self.kernel_block_stride, *kv_cache.stride()[1:]),
        )

    def get_metadata_builder(self, ubatch_id: int = 0) -> AttentionMetadataBuilder:
        assert len(self.metadata_builders) > ubatch_id
        return self.metadata_builders[ubatch_id]

    @property
    def supports_draft_decode_metadata_update(self) -> bool:
        return self.get_metadata_builder().supports_draft_decode_metadata_update

    def update_draft_decode_metadata(
        self,
        attn_metadata: Mapping[str, Any],
    ) -> None:
        metadata = attn_metadata[self.layer_names[0]]
        self.get_metadata_builder().update_draft_decode_metadata(metadata)


def map_kv_caches_to_kernel_blocks(
    kv_caches: dict[str, torch.Tensor], attn_groups: Iterable[AttentionGroup]
) -> dict[str, torch.Tensor]:
    """The views to bind to layers. The manager views in ``kv_caches`` stay for
    KV connectors."""
    mapped = dict(kv_caches)
    for group in attn_groups:
        if group.kernel_block_size is not None:
            mapped.update(
                (name, group.map_kv_cache_to_kernel_blocks(kv_caches[name]))
                for name in group.layer_names
                if name in kv_caches
            )
    return mapped


def customize_attention_spec(
    backend: type[AttentionBackend], spec: AttentionSpec
) -> AttentionSpec:
    """The backend's spec. Blocks larger than every kernel block size a backend
    takes are split into kernel blocks, which packed blocks must keep whole."""
    spec = backend.customize_spec(spec)
    sizes = backend.get_supported_kernel_block_sizes(spec)
    fixed_sizes = [s for s in sizes if isinstance(s, int)]
    if spec.block_stride_alignment is None and sizes and fixed_sizes == sizes:
        spec = replace(spec, block_stride_alignment=MultipleOf(max(fixed_sizes)))
    return spec


def _block_size_is_supported(
    backends: list[type[AttentionBackend]],
    block_size: int,
    specs: Sequence[KVCacheSpec | None],
) -> bool:
    """Check if the block size is supported by all backends.

    An exact ``int`` declaration must match exactly; a ``MultipleOf``
    declaration accepts any multiple of its base.
    """
    for backend, spec in zip(backends, specs, strict=True):
        is_supported = False
        for supported_size in backend.get_supported_kernel_block_sizes(spec):
            if isinstance(supported_size, int):
                if block_size == supported_size:
                    is_supported = True
            elif isinstance(supported_size, MultipleOf):
                if block_size % supported_size.base == 0:
                    is_supported = True
            else:
                raise ValueError(f"Unknown supported size: {supported_size}")
        if not is_supported:
            return False
    return True


def select_common_block_size(
    kv_manager_block_size: int,
    backends: list[type[AttentionBackend]],
    kv_cache_specs: Sequence[KVCacheSpec | None] | None = None,
) -> int:
    """Select a block size that is supported by all backends and is a factor of
    kv_manager_block_size.

    If kv_manager_block_size is supported by all backends, return it directly.
    Otherwise, return the max supported size.

    Args:
        kv_manager_block_size: Block size of KV cache.
        backends: List of attention backend classes.
        kv_cache_specs: Per-backend cache specs for spec-specific constraints.

    Returns:
        The selected block size.

    Raises:
        ValueError: If no valid block size found.

    """
    specs = kv_cache_specs or [None] * len(backends)
    if _block_size_is_supported(backends, kv_manager_block_size, specs):
        return kv_manager_block_size

    # MultipleOf constraints also accept the manager size if they accept a divisor.
    # Any remaining candidate must therefore be an explicit size from a backend.
    candidates = {
        size
        for backend, spec in zip(backends, specs, strict=True)
        for size in backend.get_supported_kernel_block_sizes(spec)
        if isinstance(size, int) and kv_manager_block_size % size == 0
    }

    for size in sorted(candidates, reverse=True):
        if _block_size_is_supported(backends, size, specs):
            return size
    raise ValueError(
        f"No common block size for {kv_manager_block_size} ("
        + "; ".join(
            f"{b.get_name()}: {b.get_supported_kernel_block_sizes(s)}"
            for b, s in zip(backends, specs, strict=True)
        )
        + ")."
    )


def allocate_kv_cache(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
    layout: KVCacheLayout,
    kernel_block_sizes: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Allocate the KV cache and view it as ``[B, H, N, C]`` per layer.

    Every KVCacheTensor places its layers in the same backing allocation: layer ``l`` of
    block ``b`` starts at ``offset + l * layer_stride + b * block_stride``. Cache
    groups overlay each other, so tensors may address the same bytes.
    """
    if not kv_cache_config.kv_cache_tensors:
        return {}

    sizes = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
    assert len(sizes) == 1, "KV cache tensors must share one backing allocation."
    raw_size = sizes.pop()
    # wvSplitKrc's process-lifetime static workspaces (csrc/rocm/skinny_gemms.cu)
    # are created lazily on the first qualifying GEMM. Force that now, before
    # the giant backing allocation below: if one landed in this segment's
    # rounding tail it would pin the whole segment at engine shutdown.
    if current_platform.is_rocm():
        warmup_rocm_skinny_gemm_workspaces(device)
        # Pad to the page granularity MoRIIO needs to register the shared
        # backing as a single RDMA memory region. Other platforms keep the
        # exact-size allocation (see #53974), so anything reading
        # storage.nbytes() has to tolerate the tail on ROCm alone.
        page_size = 4096
        buf_size = ((raw_size + page_size - 1) // page_size) * page_size
    else:
        buf_size = raw_size
    buf = torch.zeros(buf_size, dtype=torch.int8, device=device)

    kv_caches: dict[str, torch.Tensor] = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        layer_name = tensor.layers[0]
        group_id, group = next(
            (group_id, group)
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if layer_name in group.layer_names
        )
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.kv_cache_specs[layer_name]

        if not spec.has_layer_views:
            kv_caches.update((name, buf) for name in tensor.layers)
            continue

        num_blocks = kv_cache_config.num_blocks_of(tensor)
        kernel_block_size = None
        if kernel_block_sizes is not None and group_id < len(kernel_block_sizes):
            kernel_block_size = kernel_block_sizes[group_id]
        views = create_kv_cache_views(
            buf,
            spec,
            num_blocks,
            layout,
            tensor,
            kernel_block_size=kernel_block_size,
        )
        kv_caches.update(zip(tensor.layers, views))
    return kv_caches


def prepare_kernel_block_sizes(
    kv_cache_config: KVCacheConfig, attn_groups: list[list[AttentionGroup]]
) -> list[int]:
    """Generate kernel_block_sizes that matches each block_size.

    For attention backends that support virtual block splitting,
    use the supported block sizes from the backend.
    For other backends (like Mamba), use the same block size (no splitting).

    Args:
        kv_cache_config: The KV cache configuration.
        attn_groups: Attention groups indexed by KV cache group id.

    Returns:
        List of kernel block sizes for each cache group.

    """
    kernel_block_sizes = []
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # pick an arbitrary one to dispatch.
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            continue
        if not kv_cache_spec.has_layer_views:
            kernel_block_sizes.append(kv_cache_spec.block_size)
        elif isinstance(kv_cache_spec, AttentionSpec):
            # This is an attention backend that supports virtual block splitting.
            kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
            group_backends = [g.backend for g in attn_groups[kv_cache_gid]]
            group_specs = [g.kv_cache_spec for g in attn_groups[kv_cache_gid]]
            # Block-outermost packing puts other layers' pages between the
            # blocks, so they are never split group-wide.
            selected_kernel_size = (
                kv_manager_block_size
                if any(
                    t.block_stride > t.layer_stride
                    for t in kv_cache_config.kv_cache_tensors
                    if not set(t.layers).isdisjoint(kv_cache_group.layer_names)
                )
                else select_common_block_size(
                    kv_manager_block_size, group_backends, group_specs
                )
            )
            kernel_block_sizes.append(selected_kernel_size)
        elif isinstance(kv_cache_spec, MambaSpec):
            # This is likely Mamba or other non-attention cache, no splitting.
            kernel_block_sizes.append(kv_cache_spec.block_size)
        else:
            raise NotImplementedError(
                f"unknown kv cache spec {kv_cache_group.kv_cache_spec}"
            )
    return kernel_block_sizes


def sanity_check_mm_encoder_outputs(
    mm_embeddings: MultiModalEmbeddings,
    expected_num_items: int,
) -> None:
    """Perform sanity checks for the result of
    [`vllm.model_executor.models.SupportsMultiModal.embed_multimodal`][].
    """
    assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (
        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "
        f"or a single 3D tensor, but got {type(mm_embeddings)} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )

    assert len(mm_embeddings) == expected_num_items, (
        "Expected number of multimodal embeddings to match number of "
        f"input items: {expected_num_items}, but got {len(mm_embeddings)=} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )

    assert all(e.ndim == 2 for e in mm_embeddings), (
        "Expected multimodal embeddings to be a sequence of 2D tensors, "
        f"but got tensors with shapes {[e.shape for e in mm_embeddings]} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )


def request_memory(
    init_snapshot: MemorySnapshot,
    cache_config: CacheConfig,
    external_weight_memory: int = 0,
) -> int:
    """Calculate the amount of memory required by vLLM, then validate
    that the current amount of free memory is sufficient for that.
    """
    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )

    if external_weight_memory > 0:
        engine_memory = requested_memory - external_weight_memory
        if engine_memory <= 0:
            raise ValueError(
                f"Externally held weights on device {init_snapshot.device_} "
                f"({format_gib(external_weight_memory)}/"
                f"{format_gib(init_snapshot.total_memory)} GiB) exceed the "
                "desired GPU memory utilization "
                f"({cache_config.gpu_memory_utilization}, "
                f"{format_gib(requested_memory)} GiB). Increase GPU memory "
                "utilization or reduce GPU memory used by other processes."
            )
        if init_snapshot.free_memory < engine_memory:
            raise ValueError(
                f"Free memory on device {init_snapshot.device_} "
                f"({format_gib(init_snapshot.free_memory)}/"
                f"{format_gib(init_snapshot.total_memory)} GiB) on startup "
                "is less than the engine's budget after excluding "
                "external process's weights "
                f"({cache_config.gpu_memory_utilization}, "
                f"{format_gib(engine_memory)} GiB). Decrease GPU memory "
                "utilization or reduce GPU memory used by other processes."
            )
        logger.info_once(
            "Weights are held outside this process: of the %s GiB "
            "utilization budget, %s GiB is externally held and "
            "%s GiB remains for the engine's own allocations.",
            format_gib(requested_memory),
            format_gib(external_weight_memory),
            format_gib(engine_memory),
        )
        return engine_memory

    if init_snapshot.free_memory < requested_memory:
        raise ValueError(
            f"Free memory on device {init_snapshot.device_} "
            f"({format_gib(init_snapshot.free_memory)}/"
            f"{format_gib(init_snapshot.total_memory)} GiB) on startup "
            f"is less than desired GPU memory utilization "
            f"({cache_config.gpu_memory_utilization}, "
            f"{format_gib(requested_memory)} GiB). Decrease GPU memory "
            f"utilization or reduce GPU memory used by other processes."
        )

    return requested_memory


def add_kv_sharing_layers_to_kv_cache_groups(
    shared_kv_cache_layers: dict[str, str],
    kv_cache_groups: list[KVCacheGroupSpec],
    runner_only_attn_layers: set[str] | None = None,
) -> None:
    """Sets up KV cache sharing by reusing the allocated KV caches in `kv_caches`
    for layers that do not allocate its own KV cache, based on the mapping in
    `shared_kv_cache_layers`. Adds these layers to the corresponding KV cache
    group, which is needed to ensure that attention metadata is assigned later.

    Args:
        shared_kv_cache_layers: Layer pairings for cross-layer KV sharing.
            If an Attention layer `layer_name` is in the keys of this dict, it
            means this layer will perform attention using the keys and values
            from the KV cache of `shared_kv_cache_layers[layer_name]`.
        kv_cache_groups: The KV cache groups of the model.
        runner_only_attn_layers: Attention layers handled by the runner only,
            which are excluded from the KV cache groups.

    """
    if not shared_kv_cache_layers:
        return

    layer_to_kv_cache_group: dict[str, KVCacheGroupSpec] = {}
    for kv_cache_group in kv_cache_groups:
        for layer_name in kv_cache_group.layer_names:
            layer_to_kv_cache_group[layer_name] = kv_cache_group

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        tgt_kv_cache_group = layer_to_kv_cache_group[target_layer_name]
        tgt_kv_cache_group.layer_names.append(layer_name)

        if runner_only_attn_layers is not None:
            runner_only_attn_layers.add(layer_name)


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Any],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
    kv_cache_groups: Sequence[KVCacheGroupSpec] | None = None,
    layer_kv_caches: dict[str, torch.Tensor] | None = None,
) -> None:
    """Bind the allocated KV cache to both ModelRunner and forward context so
    that the KV cache can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's kv cache list (`runner_kv_caches`) with
         kv_caches.
      2) Associates each attention layer in the `forward_context` with its
         corresponding KV cache in kv_caches.

    Args:
        kv_caches: The allocated kv_caches with layer names as keys.
        num_attn_module: Number of attention modules per layer entry.
        forward_context: The global forward context containing all Attention
            layers with layer names as keys.
        runner_kv_caches: The kv_cache declared by ModelRunner.
        kv_cache_groups: The KV cache groups of the model, used to resolve
            layers that share a KV cache.
        layer_kv_caches: The views to bind to layers when they differ from
            kv_caches (e.g. kernel-block views). The runner keeps kv_caches,
            which are addressed by scheduler block IDs.

    """
    # Bind kv_caches to ModelRunner
    assert len(runner_kv_caches) == 0

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            # One typical case is encoder-decoder model, e.g., bart.
            # The cross attention and self attention in the same decoder layer
            # has different layer_name but the same layer_index.

            # TODO - analyze where runner_kv_caches is used and the right
            # way to ensure it properly reflects multiple attention layers
            # in the same decoder block.
            current_platform.check_runner_kv_caches_multi_layer()
        for layer_name in layer_names:
            runner_kv_caches.append(kv_caches[layer_name])

    bind_kv_cache_to_layers(
        layer_kv_caches or kv_caches, forward_context, num_attn_module, kv_cache_groups
    )


def bind_kv_cache_to_layers(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    num_attn_module: int = 1,
    kv_cache_groups: Sequence[KVCacheGroupSpec] | None = None,
) -> None:
    """Bind layer caches and share ReplaySSM trackers in model-layer order."""
    # Bind kv_caches to forward context. Each layer's bind_kv_cache unpacks
    # its raw allocation into the per-layer view(s) it needs (e.g. Mamba
    # splits conv/ssm), so the kv_caches dict can hold a single tensor per
    # layer for the KV connector to register.
    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].bind_kv_cache(kv_cache)

    # Some model-specific state is physically aligned with an owner KV cache
    # but intentionally omitted from the generic KV specs. Bind it only after
    # every regular KV cache is available.
    for _, layer in _iter_unique_layers(forward_context, KVCacheSideStorageLayer):
        layer.bind_kv_cache_side_storage(forward_context)

    ordered_layer_names = sorted(
        kv_caches, key=lambda name: extract_layer_index(name, num_attn_module)
    )
    share_replayssm_ring_trackers(ordered_layer_names, forward_context, kv_cache_groups)


def clear_layer_kv_caches(layers: Iterable[Any]) -> None:
    """Detach the KV/state cache tensors installed by bind_kv_cache().

    The model object can outlive the runner (e.g. LLMEngine's finalizer keeps
    it reachable until engine deletion), so dropping the runner's references
    alone does not release the KV cache memory on teardown paths.
    """
    for layer in layers:
        if not hasattr(layer, "kv_cache"):
            continue
        kv_cache = layer.kv_cache
        layer.kv_cache = torch.tensor([]) if isinstance(kv_cache, torch.Tensor) else []
        # Clean up quantized KV cache scale views
        # (int8_per_token_head, fp8_per_token_head)
        if hasattr(layer, "impl"):
            if hasattr(layer.impl, "_k_scale_cache"):
                layer.impl._k_scale_cache = None
            if hasattr(layer.impl, "_v_scale_cache"):
                layer.impl._v_scale_cache = None


def copy_kv_cache_blocks_inplace(
    kv_caches: Iterable[torch.Tensor],
    num_blocks: int,
    kv_cache_block_copies: Sequence[KVCacheBlockCopy],
) -> None:
    if not kv_cache_block_copies:
        return

    indices_np = np.array(kv_cache_block_copies, dtype=np.int64)
    indices: torch.Tensor | None = None
    seen: set[tuple[torch.device, int]] = set()
    copied_storages: set[tuple[torch.device, int]] = set()
    for cache in kv_caches:
        # Layers sharing KV (cross-layer sharing) alias the same view; copy it
        # once. data_ptr distinguishes per-layer views of a shared allocation.
        key = (cache.device, cache.data_ptr())
        if key in seen:
            continue
        seen.add(key)

        if indices is None:
            indices = async_tensor_h2d(indices_np, device=cache.device)
        assert cache.device == indices.device
        src, dst = indices.unbind(dim=1)

        kernel_blocks_per_block, remainder = divmod(cache.shape[0], num_blocks)
        assert remainder == 0, (
            f"{cache.shape[0]} kernel blocks not divisible by "
            f"{num_blocks} scheduler blocks"
        )
        storage = cache.untyped_storage()
        storage_key = (cache.device, storage.data_ptr())
        scheduler_block_stride = (
            cache.stride(0) * cache.element_size() * kernel_blocks_per_block
        )
        if storage.nbytes() == num_blocks * scheduler_block_stride:
            if storage_key in copied_storages:
                continue
            copied_storages.add(storage_key)
            blocks = torch.empty(0, dtype=torch.uint8, device=cache.device)
            blocks.set_(storage)
            blocks = blocks.view(num_blocks, -1)
        else:
            # Fold virtual block splitting into the shape so that dim 0 counts
            # scheduler blocks; unflatten of dim 0 is always a view.
            blocks = cache.unflatten(0, (num_blocks, kernel_blocks_per_block))
        blocks[dst] = blocks[src]


def is_uniform_query_len(num_reqs: int, num_tokens: int, max_query_len: int) -> bool:
    """Whether every request in the batch has the same query length.

    Shape test only; use ``get_uniform_decode_token_count`` to classify a
    scheduled batch, since a prompt chunk can have a decode batch's shape.
    """
    return num_reqs > 0 and num_tokens == max_query_len * num_reqs


def get_uniform_decode_token_count(
    num_reqs: int, num_tokens: int, max_query_len: int, decode_graph_eligible: bool
) -> int | None:
    """Per-request token count of a uniform decode batch, or None."""
    if decode_graph_eligible and is_uniform_query_len(
        num_reqs, num_tokens, max_query_len
    ):
        return max_query_len
    return None


def release_kv_cache_side_storage_blocks(
    static_forward_context: dict[str, Any],
    kv_cache_groups: Sequence[KVCacheGroupSpec],
    block_ids_by_group: Sequence[Sequence[int]],
) -> None:
    """Release request state without invalidating reusable prefix-cache payloads."""
    layer_name_to_group_id = _layer_name_to_group_id(kv_cache_groups)
    for layer_name, layer in _iter_unique_layers(
        static_forward_context, KVCacheSideStorageRequestLifecycle
    ):
        group_id = layer_name_to_group_id.get(layer.main_layer_name)
        if group_id is None or group_id >= len(block_ids_by_group):
            raise ValueError(
                "KV side-storage request release is missing its owner group: "
                f"layer={layer_name!r}, owner={layer.main_layer_name!r}."
            )
        if block_ids_by_group[group_id]:
            layer.release_kv_cache_side_storage(list(block_ids_by_group[group_id]))


def copy_kv_cache_side_storage_blocks(
    static_forward_context: dict[str, Any],
    kv_cache_groups: Sequence[KVCacheGroupSpec],
    kv_cache_block_copies_by_group: Sequence[Sequence[KVCacheBlockCopy]],
    num_blocks: int,
) -> None:
    """Apply owner KV copy-on-write mappings to aligned side storage."""
    layer_name_to_group_id = _layer_name_to_group_id(kv_cache_groups)
    for layer_name, layer in _iter_unique_layers(
        static_forward_context, KVCacheSideStorageLayer
    ):
        group_id = layer_name_to_group_id.get(layer.main_layer_name)
        if group_id is None:
            raise ValueError(
                "KV side-storage layer has no owner KV cache group: "
                f"layer={layer_name!r}, owner={layer.main_layer_name!r}"
            )
        if group_id >= len(kv_cache_block_copies_by_group):
            raise ValueError(
                f"KV side-storage copy mappings are missing the owner group {group_id}."
            )
        layer.copy_kv_cache_side_storage(
            kv_cache_block_copies_by_group[group_id],
            num_blocks,
        )


@dataclass
class EncoderTimingStats:
    """Per-request timing statistics for encoder forward pass."""

    encoder_forward_secs: float = 0.0
    """Time spent in vision encoder forward pass (seconds)."""

    num_encoder_calls: int = 0
    """Number of times encoder was called for this request."""

    def to_dict(self) -> dict[str, float | int]:
        return {
            "encoder_forward_secs": self.encoder_forward_secs,
            "num_encoder_calls": self.num_encoder_calls,
        }
