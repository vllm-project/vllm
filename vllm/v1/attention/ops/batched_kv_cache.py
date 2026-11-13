# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionType

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention import Attention


@triton.jit(
    do_not_specialize=["num_tokens", "key_layer_stride", "value_layer_stride"],
    do_not_specialize_on_alignment=["slots"],
)
def _batched_kv_cache_kernel(
    key,
    value,
    caches,
    slots,
    num_tokens,
    key_layer_stride: tl.int64,
    value_layer_stride: tl.int64,
    key_token_stride: tl.int64,
    value_token_stride: tl.int64,
    key_head_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    cache_strides: tl.constexpr,
    block_sizes: tl.constexpr,
    layer_ids: tl.constexpr,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    layer = tl.program_id(2)
    cache_address = tl.cast(caches[0], tl.int64)
    slot_address = tl.cast(slots[0], tl.int64)
    source_layer = tl.full((), layer_ids[0], tl.int64)
    block_size = tl.full((), block_sizes[0], tl.int64)
    block_stride = tl.full((), cache_strides[0][0], tl.int64)
    head_stride = tl.full((), cache_strides[0][1], tl.int64)
    token_stride = tl.full((), cache_strides[0][2], tl.int64)
    for i in tl.static_range(1, len(caches)):
        selected = layer == i
        cache_address = tl.where(selected, tl.cast(caches[i], tl.int64), cache_address)
        slot_address = tl.where(selected, tl.cast(slots[i], tl.int64), slot_address)
        source_layer = tl.where(selected, layer_ids[i], source_layer)
        block_size = tl.where(selected, block_sizes[i], block_size)
        block_stride = tl.where(selected, cache_strides[i][0], block_stride)
        head_stride = tl.where(selected, cache_strides[i][1], head_stride)
        token_stride = tl.where(selected, cache_strides[i][2], token_stride)
    cache = tl.cast(cache_address, caches[0].dtype)
    mapping = tl.cast(slot_address, slots[0].dtype)
    token = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
    slot = tl.load(mapping + token, token < num_tokens, -1).to(tl.int64)
    dim = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    head, col = dim // head_size, dim % head_size
    mask = (token[:, None] < num_tokens) & (slot[:, None] >= 0)
    mask &= dim[None, :] < num_heads * head_size
    key_offset = (
        source_layer * key_layer_stride
        + token[:, None] * key_token_stride
        + head[None, :] * key_head_stride
        + col[None, :]
    )
    value_offset = (
        source_layer * value_layer_stride
        + token[:, None] * value_token_stride
        + head[None, :] * value_head_stride
        + col[None, :]
    )
    destination = (
        slot[:, None] // block_size * block_stride
        + slot[:, None] % block_size * token_stride
        + head[None, :] * head_stride
        + col[None, :]
    )
    k = tl.load(key + key_offset, mask, 0)
    v = tl.load(value + value_offset, mask, 0)
    tl.store(cache + destination, k, mask)
    tl.store(cache + destination + head_size, v, mask)


def try_batched_kv_cache_update(
    attn_layers: list["Attention"],
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor | list[torch.Tensor | None],
) -> bool:
    """Scatter native context K/V for FlashAttention/FlashInfer draft layers.

    Cache and mapping pointers are kernel arguments, so rebinding needs no
    device pointer tables or host-to-device copies during graph capture.
    Unsupported configurations return False for the backend's cache writer.
    """
    shape = key.shape
    key_strides, value_strides = key.stride(), value.stride()
    if (
        not key.is_cuda
        or key.dtype not in (torch.float16, torch.bfloat16)
        or shape != value.shape
        or key.dtype != value.dtype
        or key.device != value.device
        or len(shape) != 4
        or key_strides[-1] != 1
        or value_strides[-1] != 1
        or shape[0] != len(attn_layers)
    ):
        return False
    mappings = (
        slot_mapping
        if isinstance(slot_mapping, (list, tuple))
        else [slot_mapping] * len(attn_layers)
    )
    if len(mappings) != len(attn_layers):
        return False
    layers, caches, slots = [], [], []
    cache_strides, block_sizes = [], []
    _, num_tokens, num_heads, head_size = shape
    for i, (attn, mapping) in enumerate(zip(attn_layers, mappings)):
        if mapping is None:
            continue
        cache = attn.kv_cache
        cache_shape, strides = cache.shape, cache.stride()
        if (
            attn.attn_backend.get_name() not in ("FLASH_ATTN", "FLASHINFER")
            or attn.attn_type
            not in (AttentionType.DECODER, AttentionType.ENCODER_DECODER)
            or attn.kv_cache_dtype not in ("auto", "float16", "bfloat16")
            or attn.kv_sharing_target_layer_name is not None
            or getattr(attn.impl, "dcp_world_size", 1) != 1
            or len(cache_shape) != 4
            or cache_shape[1] != num_heads
            or cache_shape[3] != 2 * head_size
            or strides[-1] != 1
            or cache.dtype != key.dtype
            or cache.device != key.device
            or mapping.shape != (num_tokens,)
            or mapping.stride(0) != 1
            or mapping.dtype != torch.int64
            or mapping.device != key.device
        ):
            return False
        layers.append(i)
        caches.append(cache)
        slots.append(mapping)
        cache_strides.append(strides)
        block_sizes.append(cache_shape[2])
    if not layers or num_tokens == 0:
        return True
    block_d = min(256, triton.next_power_of_2(num_heads * head_size))
    _batched_kv_cache_kernel[
        (
            triton.cdiv(num_tokens, 4),
            triton.cdiv(num_heads * head_size, block_d),
            len(layers),
        )
    ](
        key,
        value,
        tuple(caches),
        tuple(slots),
        num_tokens,
        key_strides[0],
        value_strides[0],
        key_strides[1],
        value_strides[1],
        key_strides[2],
        value_strides[2],
        tuple(cache_strides),
        tuple(block_sizes),
        tuple(layers),
        num_heads,
        head_size,
        4,
        block_d,
        num_warps=4,
    )
    return True
