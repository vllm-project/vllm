# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from math import prod
from typing import Any, cast

import torch

from vllm.config import (
    VllmConfig,
    get_layers_from_vllm_config,
    set_current_vllm_config,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import (
    AttentionGroup,
    add_kv_sharing_layers_to_kv_cache_groups,
    bind_kv_cache,
    prepare_kernel_block_sizes,
)


@dataclass(frozen=True)
class AttentionCGSupportInfo:
    min_cg_support: AttentionCGSupport = AttentionCGSupport.ALWAYS
    min_cg_attn_backend: str | None = None


def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    layer_type = cast(type[Any], AttentionLayerBase)
    attn_layers = get_layers_from_vllm_config(vllm_config, layer_type)
    for layer_name, attn_module in attn_layers.items():
        if getattr(attn_module, "kv_sharing_target_layer_name", None):
            # This layer will use KV cache of the sharing target layer.
            continue
        # Skip modules that don't need KV cache (eg encoder-only attention)
        if spec := attn_module.get_kv_cache_spec(vllm_config):
            if isinstance(spec, AttentionSpec):
                backend = attn_module.get_attn_backend()
                # indexes_kv_by_block_stride() -> get_kv_cache_stride_order() ->
                # get_kv_cache_layout() needs the current vLLM config.
                with set_current_vllm_config(vllm_config):
                    indexes = backend.indexes_kv_by_block_stride()
                spec = replace(spec, indexes_kv_by_block_stride=indexes)
            kv_cache_spec[layer_name] = spec
    return kv_cache_spec


def get_shared_kv_cache_layers(vllm_config: VllmConfig):
    attn_layers = get_layers_from_vllm_config(vllm_config, Attention)
    return {
        layer_name: kv_tgt_layer
        for layer_name, attn_module in attn_layers.items()
        if (kv_tgt_layer := attn_module.kv_sharing_target_layer_name)
    }


def init_attn_backend(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    device: torch.device,
    active_layer_names: set[str] | None = None,
) -> tuple[list[list[AttentionGroup]], AttentionCGSupportInfo, list[int]]:
    # Phase 1: discover attention groups for each kv cache group.
    attn_groups: list[list[AttentionGroup]] = []

    # Add KV-sharing layers to their target's kv cache group so they are
    # discovered alongside the target layer in Phase 1 below.
    add_kv_sharing_layers_to_kv_cache_groups(
        get_shared_kv_cache_layers(vllm_config), kv_cache_config.kv_cache_groups
    )

    # Phase 1: discover attention groups for each kv cache group.
    for kv_cache_group_id, kv_cache_group_spec in enumerate(
        kv_cache_config.kv_cache_groups
    ):
        layer_names = kv_cache_group_spec.layer_names
        if active_layer_names is not None:
            layer_names = list(active_layer_names.intersection(layer_names))

        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(vllm_config, layer_type, layer_names)

        group_map: dict[tuple[tuple[str, str], KVCacheSpec, int], AttentionGroup] = {}
        group_order: list[tuple[tuple[str, str], KVCacheSpec, int]] = []

        for layer_name in layer_names:
            attn_backend = attn_layers[layer_name].get_attn_backend()

            layer_kv_cache_spec: KVCacheSpec = kv_cache_group_spec.kv_cache_spec
            if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]

            # Split on per-rank num_heads_q so layers with different Q-head
            # counts (e.g. a spec-decode draft head and its target) get separate
            # metadata builders.
            num_heads_q = getattr(attn_layers[layer_name], "num_heads", 0)
            key = (attn_backend.full_cls_name(), layer_kv_cache_spec, num_heads_q)
            if key not in group_map:
                group_map[key] = AttentionGroup(
                    attn_backend, [layer_name], layer_kv_cache_spec, kv_cache_group_id
                )
                group_order.append(key)
            else:
                group_map[key].layer_names.append(layer_name)

        attn_groups.append([group_map[key] for key in group_order])

    # Phase 2: pick a kernel block size per kv cache group that is supported
    # by all backends within that group.
    kernel_block_sizes = prepare_kernel_block_sizes(kv_cache_config, attn_groups)

    # Phase 3: create metadata builders and determine cudagraph support.
    attn_backend_workspace: torch.Tensor | None = None
    min_cg_support = AttentionCGSupport.ALWAYS
    min_cg_attn_backend = None
    for kv_cache_group_id, groups in enumerate(attn_groups):
        kernel_block_size = None
        if kv_cache_group_id < len(kernel_block_sizes):
            kernel_block_size = kernel_block_sizes[kv_cache_group_id]
        for group in groups:
            group.create_metadata_builders(
                vllm_config=vllm_config,
                device=device,
                kernel_block_size=kernel_block_size,
                num_metadata_builders=1,
            )
            builder = group.get_metadata_builder(0)
            if attn_backend_workspace is None:
                if hasattr(builder, "_get_workspace_buffer"):
                    attn_backend_workspace = builder._get_workspace_buffer()
            else:
                if hasattr(builder, "set_workspace_buffer"):
                    builder.set_workspace_buffer(attn_backend_workspace)
            # Check cudagraph support for the attention backend
            cg_support = builder.get_cudagraph_support(
                vllm_config,
                cast(AttentionSpec, group.kv_cache_spec),
            )
            if cg_support.value < min_cg_support.value:
                min_cg_support = cg_support
                min_cg_attn_backend = group.backend.__name__

    attn_cg_support_info = AttentionCGSupportInfo(
        min_cg_support=min_cg_support, min_cg_attn_backend=min_cg_attn_backend
    )
    return attn_groups, attn_cg_support_info, kernel_block_sizes


def _allocate_kv_cache(
    kv_cache_config: KVCacheConfig, shared_layers: dict[str, str], device: torch.device
):
    kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
    packed_backing: torch.Tensor | None = None
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        if kv_cache_tensor.block_stride > 0:
            # Allocate once; all packed tensors alias the same backing.
            if packed_backing is None:
                packed_backing = torch.zeros(
                    kv_cache_tensor.size, dtype=torch.int8, device=device
                )
            tensor = packed_backing
        else:
            tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tensor

    layer_names = set()
    for group in kv_cache_config.kv_cache_groups:
        for layer_name in group.layer_names:
            layer_names.add(layer_name)
    assert layer_names == (kv_cache_raw_tensors.keys() | shared_layers.keys()), (
        "Some layers are not correctly initialized"
    )
    return kv_cache_raw_tensors


def _reshape_attention_kv_cache(
    kv_raw_tensor: torch.Tensor,
    kv_cache_spec: AttentionSpec,
    kv_cache_shape: tuple[int, ...],
    kv_cache_stride_order: tuple[int, ...],
    num_blocks: int,
    packing: tuple[int, int] | None,
) -> torch.Tensor:
    permuted_kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
    inv_order = [
        kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
    ]
    dtype = kv_cache_spec.dtype

    if packing is not None:
        offset, block_stride = packing
        assert inv_order[0] == 0
        page_bytes = prod(kv_cache_shape[1:]) * get_dtype_size(dtype)
        kv_cache = (
            kv_raw_tensor.view(-1, block_stride)[:, offset : offset + page_bytes]
            .view(dtype)
            .view(permuted_kv_cache_shape)
        )
    elif kv_cache_spec.page_size_padded is not None:
        # Use a strided view to skip the padding between physical pages.
        #
        # Only num-blocks-first layouts are supported (the block dimension is
        # dim 0 of the unpermuted shape). kv-first layouts such as ROCm's
        # ``(2, num_blocks, ...)`` are intentionally not supported here. For a
        # num-blocks-first layout the only stride that must change is the block
        # stride: every other (contiguous) stride already steps within the
        # unpadded region of a page, so no further adjustment is needed.
        assert kv_cache_shape[0] == num_blocks, (
            "Padded KV pages require a num-blocks-first KV cache layout (got "
            f"shape {kv_cache_shape} with num_blocks={num_blocks}); "
            "kv-first layouts are not supported."
        )
        dtype_size = get_dtype_size(kv_cache_spec.dtype)
        page_stride = kv_cache_spec.page_size_bytes // dtype_size

        num_blocks_dim = inv_order[0]
        strides = list(torch.empty(permuted_kv_cache_shape).stride())
        strides[num_blocks_dim] = page_stride

        kv_cache = torch.as_strided(
            kv_raw_tensor.view(dtype),
            size=permuted_kv_cache_shape,
            stride=tuple(strides),
        )
    else:
        # No padding — safe to use a contiguous view.
        kv_cache = kv_raw_tensor.view(dtype).view(permuted_kv_cache_shape)

    return kv_cache.permute(*inv_order)


def _reshape_kv_cache(
    attn_groups: Sequence[AttentionGroup],
    kv_cache_raw_tensors: dict[str, torch.Tensor],
    cache_dtype: str,
    kernel_block_sizes: list[int],
    shared_kv_cache_layers: dict[str, str],
    kv_cache_config: "KVCacheConfig | None" = None,
) -> dict[str, Any]:
    kv_caches: dict[str, Any] = {}
    has_attn, has_mamba = False, False

    layer_packing: dict[str, tuple[int, int]] = {}
    if kv_cache_config is not None:
        for kv_tensor in kv_cache_config.kv_cache_tensors:
            if kv_tensor.block_stride > 0:
                for ln in kv_tensor.shared_by:
                    layer_packing[ln] = (kv_tensor.offset, kv_tensor.block_stride)

    for group in attn_groups:
        if group.kv_cache_group_id >= len(kernel_block_sizes):
            continue

        kv_cache_spec = group.kv_cache_spec
        if kv_cache_spec.storage_block_size != kv_cache_spec.block_size:
            # use storage_block_size as the kernel block size for groups
            # that apply a compression on block size (eg. DeepSeek V4).
            kernel_block_size = kv_cache_spec.storage_block_size
        else:
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]

        for layer_name in group.layer_names:
            if layer_name in shared_kv_cache_layers:
                # Shared layer — tensor will be aliased to its target later.
                continue

            kv_raw_tensor = kv_cache_raw_tensors[layer_name]
            packing = layer_packing.get(layer_name)
            if packing is not None:
                _, blk_stride = packing
                num_blocks = kv_raw_tensor.numel() // blk_stride
            else:
                assert kv_raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = kv_raw_tensor.numel() // kv_cache_spec.page_size_bytes

            if isinstance(kv_cache_spec, AttentionSpec):
                has_attn = True
                # Use storage_block_size: it equals block_size for uncompressed
                # specs but is smaller for compressed ones (DeepSeek V4), which
                # store block_size tokens in block_size // compress_ratio slots.
                num_blocks_per_kv_block = (
                    kv_cache_spec.storage_block_size // kernel_block_size
                )
                kernel_num_blocks = num_blocks * num_blocks_per_kv_block
                kv_cache_shape = group.backend.get_kv_cache_shape(
                    kernel_num_blocks,
                    kernel_block_size,
                    kv_cache_spec.num_kv_heads,
                    kv_cache_spec.head_size,
                    cache_dtype_str=cache_dtype,
                )

                # FIXME(woosuk): Add kv_cache_stride_order to all attention backends.
                try:
                    kv_cache_stride_order = group.backend.get_kv_cache_stride_order()
                    assert len(kv_cache_stride_order) == len(kv_cache_shape)
                except (AttributeError, NotImplementedError):
                    kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

                kv_caches[layer_name] = _reshape_attention_kv_cache(
                    kv_raw_tensor,
                    kv_cache_spec,
                    kv_cache_shape,
                    kv_cache_stride_order,
                    kernel_num_blocks,
                    packing,
                )

            elif isinstance(kv_cache_spec, MambaSpec):
                has_mamba = True
                state_tensors = []
                storage_offset_bytes = 0
                for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                    dtype_size = get_dtype_size(dtype)
                    num_element_per_page = kv_cache_spec.page_size_bytes // dtype_size
                    target_shape = (num_blocks, *shape)
                    stride = torch.empty(target_shape).stride()
                    target_stride = (num_element_per_page, *stride[1:])
                    assert storage_offset_bytes % dtype_size == 0
                    tensor = torch.as_strided(
                        kv_raw_tensor.view(dtype),
                        size=target_shape,
                        stride=target_stride,
                        storage_offset=storage_offset_bytes // dtype_size,
                    )
                    state_tensors.append(tensor)
                    storage_offset_bytes += stride[0] * dtype_size
                kv_caches[layer_name] = state_tensors
            else:
                raise NotImplementedError(
                    f"Unsupported KV cache spec type: {type(kv_cache_spec)}"
                )

    if has_attn and has_mamba:
        _update_hybrid_attention_layout(
            attn_groups=attn_groups,
            kv_caches=kv_caches,
            kernel_block_sizes=kernel_block_sizes,
            cache_dtype=cache_dtype,
        )

    # Map any sharing layers to their target layer's KV cache.
    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        kv_caches[layer_name] = kv_caches[target_layer_name]

    return kv_caches


def _update_hybrid_attention_layout(
    attn_groups: Iterable[AttentionGroup],
    kv_caches: dict[str, Any],
    kernel_block_sizes: list[int],
    cache_dtype: str,
) -> None:
    for group in attn_groups:
        if group.kv_cache_group_id >= len(kernel_block_sizes):
            continue

        kv_cache_spec = group.kv_cache_spec
        if not isinstance(kv_cache_spec, AttentionSpec):
            continue
        block_dim = group.backend.get_kv_cache_block_dim(
            kernel_block_sizes[group.kv_cache_group_id],
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            cache_dtype_str=cache_dtype,
        )
        # if the first dim of the kvcache's layout is already num_blocks, continue
        if block_dim == 0:
            continue

        assert block_dim == 1, (
            "Expected the dim `num_blocks` at the second dim when updating"
            " the kvcache's layout of full attention layer"
        )

        for layer_name in group.layer_names:
            if layer_name not in kv_caches:
                # Shared layer — will be aliased to its target after this pass.
                continue

            kv_cache = kv_caches[layer_name]
            if kv_cache.shape[0] == 2:
                assert kv_cache.shape[1] != 2, (
                    f"Cannot determine layout for tensor of shape {kv_cache.shape}"
                )
                hidden_size = kv_cache.shape[2:].numel()
                kv_cache.as_strided_(
                    size=kv_cache.shape,
                    stride=(
                        hidden_size,
                        2 * hidden_size,
                        *kv_cache.stride()[2:],
                    ),
                )


def init_kv_cache(
    runner_kv_caches: list[torch.Tensor | list[torch.Tensor]],
    forward_context: dict[str, Any],
    kv_cache_config: KVCacheConfig,
    attn_groups: list[list[AttentionGroup]],
    device: torch.device,
    cache_dtype: str,
    kernel_block_sizes: list[int],
    vllm_config: VllmConfig,
) -> dict[str, Any]:
    shared_kv_cache_layers = get_shared_kv_cache_layers(vllm_config)
    kv_cache_raw_tensors = _allocate_kv_cache(
        kv_cache_config, shared_kv_cache_layers, device
    )
    flattened_attn_groups = list(group for groups in attn_groups for group in groups)
    kv_caches = _reshape_kv_cache(
        attn_groups=flattened_attn_groups,
        kv_cache_raw_tensors=kv_cache_raw_tensors,
        kernel_block_sizes=kernel_block_sizes,
        cache_dtype=cache_dtype,
        shared_kv_cache_layers=shared_kv_cache_layers,
        kv_cache_config=kv_cache_config,
    )
    bind_kv_cache(kv_caches, forward_context, runner_kv_caches)
    return kv_caches


def build_slot_mappings_by_layer(
    slot_mappings: torch.Tensor, kv_cache_config: KVCacheConfig
) -> dict[str, torch.Tensor]:
    slot_mappings_by_layer: dict[str, torch.Tensor] = {}
    kv_cache_groups = kv_cache_config.kv_cache_groups
    for slot_mapping, kv_cache_group in zip(slot_mappings, kv_cache_groups):
        for layer_name in kv_cache_group.layer_names:
            slot_mappings_by_layer[layer_name] = slot_mapping
    return slot_mappings_by_layer


def build_attn_metadata(
    attn_groups: list[list[AttentionGroup]],
    num_reqs: int,
    num_tokens: int,
    query_start_loc_gpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    max_query_len: int,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    block_tables: Sequence[torch.Tensor],
    slot_mappings: torch.Tensor,
    kv_cache_config: KVCacheConfig,
    seq_lens_cpu_upper_bound: torch.Tensor | None = None,
    dcp_local_seq_lens: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    for_cudagraph_capture: bool = False,
    causal: bool = True,
    rswa_prefix_lens: torch.Tensor | None = None,
) -> dict[str, Any]:
    seq_lens = seq_lens[:num_reqs]
    if dcp_local_seq_lens is not None:
        dcp_local_seq_lens = dcp_local_seq_lens[:num_reqs]
    if seq_lens_cpu_upper_bound is not None:
        seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound[:num_reqs]

    attn_metadata: dict[str, Any] = {}
    num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
    for i in range(num_kv_cache_groups):
        block_table = block_tables[i]
        slot_mapping = slot_mappings[i]

        common_attn_metadata_extra_kwargs = (
            model_specific_attn_metadata.get_extra_common_attn_kwargs(i, num_reqs)
            if model_specific_attn_metadata is not None
            else {}
        )
        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            max_seq_len=max_seq_len,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=causal,
            dcp_local_seq_lens=dcp_local_seq_lens,
            positions=positions,
            rswa_prefix_lens=rswa_prefix_lens,
            **common_attn_metadata_extra_kwargs,
        )

        for attn_group in attn_groups[i]:
            attn_metadata_builder = attn_group.get_metadata_builder(0)
            if for_cudagraph_capture:
                metadata = attn_metadata_builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
            else:
                attn_metadata_extra_kwargs = (
                    model_specific_attn_metadata.get_extra_attn_kwargs(
                        attn_metadata_builder,
                        num_reqs,
                    )
                    if model_specific_attn_metadata is not None
                    else {}
                )
                metadata = attn_metadata_builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common_attn_metadata,
                    **attn_metadata_extra_kwargs,
                )
            for layer_name in attn_group.layer_names:
                attn_metadata[layer_name] = metadata
    return attn_metadata
