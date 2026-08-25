# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product as iprod
from typing import Any

import numpy as np
import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    select_common_block_size_from_constraints,
)
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
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
        runner_only_attn_layers: set[str] | None = None,
        zeroing_group_ids: set[int] | None = None,
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
        When ``zeroing_group_ids`` is provided, groups in other physical
        block-pool domains are excluded because their numeric block IDs may
        overlap.
        """
        self.device = device
        self._meta: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int] | None
        ) = None

        if runner_only_attn_layers is None:
            runner_only_attn_layers = set()
        # Overlaid layers (packed layouts) share a base address but may have
        # different page sizes; keep the widest span per address so newly
        # allocated blocks are fully zeroed for every overlaying group.
        seen_ptrs: dict[int, int] = {}
        seg_addrs: list[int] = []
        seg_block_strides: list[int] = []
        seg_page_sizes: list[int] = []

        for group in attn_groups_iter:
            if (
                zeroing_group_ids is not None
                and group.kv_cache_group_id not in zeroing_group_ids
            ):
                continue
            spec = group.kv_cache_spec
            if not isinstance(spec, AttentionSpec):
                continue
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
            assert spec.block_size % kernel_bs == 0
            ratio = spec.block_size // kernel_bs

            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                kv = static_forward_context[layer_name].kv_cache
                if not isinstance(kv, torch.Tensor):
                    continue
                dp = kv.data_ptr()

                el = kv.element_size()
                block_stride_bytes = kv.stride(0) * el
                assert block_stride_bytes % 4 == 0
                assert kv.shape[0] % ratio == 0
                outer_dims = [
                    d
                    for d in range(1, kv.ndim)
                    if kv.stride(d) * el > block_stride_bytes
                ]
                outer_strides = [kv.stride(d) * el for d in outer_dims]
                inner_dims = [d for d in range(1, kv.ndim) if d not in outer_dims]
                kernel_page_bytes = el + sum(
                    (kv.shape[d] - 1) * kv.stride(d) * el for d in inner_dims
                )
                assert kernel_page_bytes % 4 == 0
                logical_block_stride_bytes = block_stride_bytes * ratio
                for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):
                    off_bytes = sum(i * s for i, s in zip(outer, outer_strides))
                    assert (dp + off_bytes) % 4 == 0
                    for virtual_index in range(ratio):
                        addr = dp + off_bytes + virtual_index * block_stride_bytes
                        if (idx := seen_ptrs.get(addr)) is not None:
                            assert (
                                seg_block_strides[idx]
                                == logical_block_stride_bytes // 4
                            )
                            seg_page_sizes[idx] = max(
                                seg_page_sizes[idx], kernel_page_bytes // 4
                            )
                            continue
                        seen_ptrs[addr] = len(seg_addrs)
                        seg_addrs.append(addr)
                        seg_block_strides.append(logical_block_stride_bytes // 4)
                        seg_page_sizes.append(kernel_page_bytes // 4)

        if not seg_addrs:
            self._meta = None
            return

        max_page_size_el = max(seg_page_sizes)
        blk_size = min(1 << (max_page_size_el - 1).bit_length(), 1024)
        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
            torch.tensor(seg_block_strides, dtype=torch.int64, device=self.device),
            torch.tensor(seg_page_sizes, dtype=torch.int64, device=self.device),
            (max_page_size_el + blk_size - 1) // blk_size,
            blk_size,
            len(seg_addrs),
        )

    def zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        if not block_ids or self._meta is None:
            return
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

    def warmup(self, num_kv_blocks: int) -> None:
        """JIT-compile the zeroing kernel before the first real request."""
        if num_kv_blocks > 0:
            self.zero_block_ids([0])


def build_kv_block_zeroers(
    *,
    device: torch.device,
    attn_groups: list[list["AttentionGroup"]],
    kernel_block_sizes: list[int],
    static_forward_context: dict[str, Any],
    kv_cache_config: KVCacheConfig,
    runner_only_attn_layers: set[str] | None = None,
) -> dict[int, KVBlockZeroer]:
    return {
        pool_id: KVBlockZeroer(
            device,
            attn_groups_iter=(group for groups in attn_groups for group in groups),
            kernel_block_sizes=kernel_block_sizes,
            static_forward_context=static_forward_context,
            runner_only_attn_layers=runner_only_attn_layers,
            zeroing_group_ids={
                group_id
                for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
                if group.block_pool_id == pool_id
            },
        )
        for pool_id in kv_cache_config.zeroing_block_pool_ids
    }


@dataclass
class AttentionGroup:
    backend: type[AttentionBackend]
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    kv_cache_group_id: int
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
        kv_cache_spec_builder = (
            self.kv_cache_spec.copy_with_new_block_size(kernel_block_size)
            if kernel_block_size is not None
            else self.kv_cache_spec
        )
        builder_cls = self.backend.get_builder_cls()
        builder_kwargs = {}
        if builder_cls.requires_block_table_width:
            max_num_blocks = self.kv_cache_spec.max_num_blocks_per_req(
                vllm_config, vllm_config.model_config.max_model_len
            )
            builder_kwargs["block_table_width"] = get_block_table_width(
                max_num_blocks, self.kv_cache_spec.block_size, kernel_block_size
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


def select_common_block_size(
    kv_manager_block_size: int,
    backends: list[type[AttentionBackend]],
) -> int:
    """
    Select a block size that is supported by all backends and is a factor of
    kv_manager_block_size.

    If kv_manager_block_size is supported by all backends, return it directly.
    Otherwise, return the max supported size.

    Args:
        kv_manager_block_size: Block size of KV cache.
        backends: List of attention backend classes.

    Returns:
        The selected block size.

    Raises:
        ValueError: If no valid block size found.
    """

    if not backends:
        return kv_manager_block_size

    return select_common_block_size_from_constraints(
        kv_manager_block_size,
        [backend.get_supported_kernel_block_sizes() for backend in backends],
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
    page_size = 4096
    buf = torch.zeros(
        ((raw_size + page_size - 1) // page_size) * page_size,
        dtype=torch.int8,
        device=device,
    )

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

        num_blocks = kv_cache_config.num_blocks
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
    """
    Generate kernel_block_sizes that matches each block_size.

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
        if isinstance(kv_cache_spec, AttentionSpec):
            # This is an attention backend that supports virtual block splitting.
            kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
            group_backends = [g.backend for g in attn_groups[kv_cache_gid]]
            selected_kernel_size = select_common_block_size(
                kv_manager_block_size, group_backends
            )
            kernel_block_sizes.append(selected_kernel_size)
        elif isinstance(kv_cache_spec, MambaSpec):
            # This is likely Mamba or other non-attention cache, no splitting.
            kernel_block_sizes.append(kv_cache_spec.block_size)
        elif isinstance(kv_cache_spec, (HiSparseHotSpec, HiSparseResidentSpec)):
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
    """
    Perform sanity checks for the result of
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


def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    """
    Calculate the amount of memory required by vLLM, then validate
    that the current amount of free memory is sufficient for that.
    """
    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )

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
    """
    Sets up KV cache sharing by reusing the allocated KV caches in `kv_caches`
    for layers that do not allocate its own KV cache, based on the mapping in
    `shared_kv_cache_layers`. Adds these layers to the corresponding KV cache
    group, which is needed to ensure that attention metadata is assigned later.

    Args:
        shared_kv_cache_layers: Layer pairings for cross-layer KV sharing.
            If an Attention layer `layer_name` is in the keys of this dict, it
            means this layer will perform attention using the keys and values
            from the KV cache of `shared_kv_cache_layers[layer_name]`.
        kv_cache_groups: The KV cache groups of the model.
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
    forward_context: dict[str, Attention],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """
    Bind the allocated KV cache to both ModelRunner and forward context so
    that the KV cache can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's kv cache list (`runner_kv_caches`) with
         kv_caches.
      2) Associates each attention layer in the `forward_context` with its
         corresponding KV cache in kv_caches.

    Args:
        kv_caches: The allocated kv_caches with layer names as keys.
        forward_context: The global forward context containing all Attention
            layers with layer names as keys.
        runner_kv_caches: The kv_cache declared by ModelRunner.
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

    # Bind kv_caches to forward context. Each layer's bind_kv_cache unpacks
    # its raw allocation into the per-layer view(s) it needs (e.g. Mamba
    # splits conv/ssm), so the kv_caches dict can hold a single tensor per
    # layer for the KV connector to register.
    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].bind_kv_cache(kv_cache)


class DeviceKVCacheBlockCopier:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        kv_caches: Mapping[str, torch.Tensor | list[torch.Tensor]],
    ) -> None:
        self._num_blocks_by_pool = kv_cache_config.num_blocks_by_pool
        self._caches_by_pool: dict[int, list[torch.Tensor | list[torch.Tensor]]] = (
            defaultdict(list)
        )
        for group in kv_cache_config.kv_cache_groups:
            pool_id = group.block_pool_id
            if pool_id is None:
                continue
            self._caches_by_pool[pool_id].extend(
                kv_caches[name] for name in group.layer_names if name in kv_caches
            )

    def copy(self, copies: Sequence[KVCacheBlockCopy]) -> None:
        copies_by_pool: dict[int, list[KVCacheBlockCopy]] = defaultdict(list)
        for copy in copies:
            if copy.block_pool_id is not None:
                copies_by_pool[copy.block_pool_id].append(copy)
        for pool_id, pool_copies in copies_by_pool.items():
            copy_kv_cache_blocks_inplace(
                self._caches_by_pool[pool_id],
                self._num_blocks_by_pool[pool_id],
                pool_copies,
            )


def copy_kv_cache_blocks_inplace(
    kv_caches: Iterable[torch.Tensor],
    num_blocks: int,
    kv_cache_block_copies: Sequence[KVCacheBlockCopy],
    host_write_event: torch.Event | None = None,
) -> None:
    if not kv_cache_block_copies:
        return

    indices_np = np.array(
        [(copy.src_block_id, copy.dst_block_id) for copy in kv_cache_block_copies],
        dtype=np.int64,
    )
    indices: torch.Tensor | None = None
    seen: set[tuple[torch.device, int]] = set()
    copied_storages: set[tuple[torch.device, int]] = set()
    host_writes_synchronized = False
    for cache in kv_caches:
        # Layers sharing KV (cross-layer sharing) alias the same view; copy it
        # once. data_ptr distinguishes per-layer views of a shared allocation.
        key = (cache.device, cache.data_ptr())
        if key in seen:
            continue
        seen.add(key)

        if (
            cache.device.type == "cpu"
            and host_write_event is not None
            and not host_writes_synchronized
        ):
            host_write_event.synchronize()
            host_writes_synchronized = True
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
    num_reqs: int, num_tokens: int, max_query_len: int, has_prefill: bool
) -> int | None:
    """Per-request token count of a uniform decode batch, or None."""
    if not has_prefill and is_uniform_query_len(num_reqs, num_tokens, max_query_len):
        return max_query_len
    return None


def is_residual_scattered_for_sp(
    vllm_config: VllmConfig, num_input_tokens: int
) -> bool:
    """Check if the residual tensor is scattered for sequence parallelism.

    The residual tensor is scattered across tensor parallel ranks when sequence
    parallelism and tensor parallelism is enabled. SP is only supported in
    full-graph compilation mode.
    """
    if not vllm_config.compilation_config.pass_config.enable_sp:
        return False

    tp = vllm_config.parallel_config.tensor_parallel_size

    if tp == 1:
        return False

    assert (
        vllm_config.compilation_config.use_inductor_graph_partition
        or not vllm_config.compilation_config.splitting_ops
    ), "Sequence parallelism requires full-graph compilation"

    # When sequence parallelism is enabled, we always pad num_input_tokens
    # to be a multiple of tensor_parallel_size (tp) earlier.
    assert num_input_tokens % tp == 0

    return True


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
