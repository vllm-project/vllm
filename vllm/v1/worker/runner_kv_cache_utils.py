# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch

from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
from vllm.v1.worker.utils import AttentionGroup


def reshape_kv_cache_tensors(
    attn_groups: Iterable[AttentionGroup],
    kv_cache_raw_tensors: dict[str, torch.Tensor],
    kernel_block_sizes: list[int],
    cache_dtype: str,
    skip_layer_names: set[str] | None = None,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """
    Reshape the KV cache tensors to the desired shape and dtype.

    Args:
        attn_groups: The attention groups of all the layers.
        kv_cache_raw_tensors: The KV cache buffer of each layer, with
            correct size but uninitialized shape.
        kernel_block_sizes: The kernel block sizes for each KV cache group.
        cache_dtype: The dtype of KV cache.
        skip_layer_names: Layers have no KV cache need to skip.
    Returns:
        Dict[str, torch.Tensor]: A map between layer names to their
        corresponding memory buffer for KV cache.
    """
    groups = tuple(attn_groups)
    kv_caches: dict[str, torch.Tensor | list[torch.Tensor]] = {}
    has_attn = False
    has_mamba = False

    for group in groups:
        if group.kv_cache_group_id >= len(kernel_block_sizes):
            continue

        kv_cache_spec = group.kv_cache_spec
        kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
        for layer_name in group.layer_names:
            if skip_layer_names is not None and layer_name in skip_layer_names:
                continue

            raw_tensor = kv_cache_raw_tensors[layer_name]
            assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
            num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes

            if isinstance(kv_cache_spec, AttentionSpec):
                has_attn = True
                num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
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

                kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
                inv_order = [
                    kv_cache_stride_order.index(i)
                    for i in range(len(kv_cache_stride_order))
                ]
                kv_caches[layer_name] = (
                    raw_tensor.view(kv_cache_spec.dtype)
                    .view(kv_cache_shape)
                    .permute(*inv_order)
                )
            elif isinstance(kv_cache_spec, MambaSpec):
                has_mamba = True
                state_tensors: list[torch.Tensor] = []
                storage_offset_bytes = 0
                for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                    dtype_size = get_dtype_size(dtype)
                    num_elements_per_page = kv_cache_spec.page_size_bytes // dtype_size
                    target_shape = (num_blocks, *shape)
                    stride = torch.empty(target_shape).stride()
                    target_stride = (num_elements_per_page, *stride[1:])
                    assert storage_offset_bytes % dtype_size == 0
                    tensor = torch.as_strided(
                        raw_tensor.view(dtype),
                        size=target_shape,
                        stride=target_stride,
                        storage_offset=storage_offset_bytes // dtype_size,
                    )
                    state_tensors.append(tensor)
                    storage_offset_bytes += stride[0] * dtype_size
                kv_caches[layer_name] = state_tensors
            else:
                raise NotImplementedError(
                    f"Unsupported kv cache spec type: {type(kv_cache_spec)}"
                )

    if has_attn and has_mamba:
        update_hybrid_attention_mamba_layout(
            attn_groups=groups,
            kv_caches=kv_caches,
            kernel_block_sizes=kernel_block_sizes,
            cache_dtype=cache_dtype,
        )
    return kv_caches


def update_hybrid_attention_mamba_layout(
    attn_groups: Iterable[AttentionGroup],
    kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
    kernel_block_sizes: list[int],
    cache_dtype: str,
) -> None:
    """
    Update the layout of attention layers from (2, num_blocks, ...) to
    (num_blocks, 2, ...).

    Args:
        attn_groups: The attention groups of all the layers.
        kv_caches: The KV cache buffer of each layer.
        kernel_block_sizes: The kernel block sizes for each KV cache group.
        cache_dtype: The dtype of KV cache.
    """

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
            kv_cache = kv_caches[layer_name]
            if not isinstance(kv_cache, torch.Tensor):
                continue

            if isinstance(kv_cache_spec, AttentionSpec) and kv_cache.shape[0] == 2:
                assert kv_cache.shape[1] != 2, (
                    "Fail to determine whether the layout is "
                    "(2, num_blocks, ...) or (num_blocks, 2, ...) for "
                    f"a tensor of shape {kv_cache.shape}"
                )

            hidden_size = kv_cache.shape[2:].numel()
            kv_cache.as_strided_(
                size=kv_cache.shape,
                stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
            )
