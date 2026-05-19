# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up TurboQuant decode kernels before serving requests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.turboquant_attn import (
    TurboQuantAttentionImpl,
    TurboQuantMetadata,
)
from vllm.v1.worker.workspace import is_workspace_manager_initialized

logger = init_logger(__name__)


@dataclass(frozen=True)
class _TurboQuantDecodeWarmupKey:
    num_kv_heads: int
    head_dim: int
    block_size: int
    block_table_stride: int
    num_kv_splits: int
    kv_group_size: int
    scale: float
    mse_bits: int
    key_packed_size: int
    value_quant_bits: int
    key_fp8: bool
    norm_correction: bool
    output_fp16: bool


def _iter_turboquant_attention_layers(
    model: torch.nn.Module,
) -> Iterable[tuple[Attention, TurboQuantAttentionImpl]]:
    for layer in model.modules():
        if not isinstance(layer, Attention):
            continue
        if not layer.kv_cache_dtype.startswith("turboquant_"):
            continue
        if not isinstance(layer.impl, TurboQuantAttentionImpl):
            continue
        yield layer, layer.impl


def _make_warmup_key(
    impl: TurboQuantAttentionImpl,
    *,
    block_size: int,
    block_table_stride: int,
    model_dtype: torch.dtype,
) -> _TurboQuantDecodeWarmupKey:
    return _TurboQuantDecodeWarmupKey(
        num_kv_heads=impl.num_kv_heads,
        head_dim=impl.head_size,
        block_size=block_size,
        # Triton specializes regular scalar stride arguments too. Keep the
        # synthetic block table stride equal to the runtime block table stride
        # and include it in the dedupe key so warmup covers the same variant.
        block_table_stride=block_table_stride,
        num_kv_splits=impl.max_num_kv_splits,
        kv_group_size=impl.num_kv_groups,
        scale=impl.scale,
        mse_bits=impl.tq_config.key_mse_bits,
        key_packed_size=impl.tq_config.key_packed_size,
        value_quant_bits=impl.tq_config.effective_value_quant_bits,
        key_fp8=impl.tq_config.key_fp8,
        norm_correction=impl.tq_config.norm_correction,
        output_fp16=model_dtype == torch.float16,
    )


def _warmup_turboquant_decode_layer(
    layer: Attention,
    impl: TurboQuantAttentionImpl,
    *,
    device: torch.device,
    block_size: int,
    block_table_stride: int,
    max_num_decode_tokens: int,
    model_dtype: torch.dtype,
) -> None:
    impl._ensure_on_device(layer, device)

    batch_size = max_num_decode_tokens
    query = torch.zeros(
        (batch_size, impl.num_heads, impl.head_size),
        dtype=model_dtype,
        device=device,
    )
    kv_cache = torch.zeros(
        (
            2,
            block_size,
            impl.num_kv_heads,
            impl.tq_config.slot_size_aligned,
        ),
        dtype=torch.uint8,
        device=device,
    )
    block_table = torch.zeros(
        (batch_size, block_table_stride), dtype=torch.int32, device=device
    )
    block_table[:, 0] = 1
    seq_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
    attn_metadata = TurboQuantMetadata(
        seq_lens=seq_lens,
        slot_mapping=torch.zeros(batch_size, dtype=torch.long, device=device),
        block_table=block_table,
        query_start_loc=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
        num_actual_tokens=batch_size,
        max_query_len=1,
        max_seq_len=1,
        is_prefill=False,
        num_decodes=batch_size,
        num_decode_tokens=batch_size,
    )

    # Use the runtime decode helper instead of calling the Triton launcher
    # directly. This warms both the decode kernels and the WorkspaceManager
    # allocation path before the workspace is locked after CUDA graph capture.
    impl._decode_attention(
        query=query,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        Pi=layer._tq_Pi,
        centroids=layer._tq_centroids,
        PiT=layer._tq_PiT,
        layer=layer,
    )

    if not is_workspace_manager_initialized():
        return

    # Warm the continuation-prefill path too. That path bulk-dequants cached
    # TQ KV through `_tq_full_dequant_kv`, then runs a small attention over the
    # cached prefix plus the current chunk. Use one request and one chunk token;
    # the cached prefix points at block 1, matching the decode warmup cache.
    continuation_query = torch.zeros(
        (1, impl.num_heads, impl.head_size),
        dtype=model_dtype,
        device=device,
    )
    key_chunk = torch.zeros(
        (1, impl.num_kv_heads, impl.head_size),
        dtype=model_dtype,
        device=device,
    )
    value_chunk = torch.zeros_like(key_chunk)
    continuation_block_table = torch.zeros(
        (1, block_table_stride), dtype=torch.int32, device=device
    )
    continuation_block_table[0, 0] = 1

    impl._continuation_prefill(
        layer=layer,
        query=continuation_query,
        key_chunk=key_chunk,
        val_chunk=value_chunk,
        kv_cache=kv_cache,
        block_table=continuation_block_table,
        cached_len=block_size,
        seq_len=block_size + 1,
        Pi=layer._tq_Pi,
        centroids=layer._tq_centroids,
    )


@torch.inference_mode()
def turboquant_decode_warmup(
    model: torch.nn.Module,
    *,
    device: torch.device,
    block_size: int,
    block_table_stride: int,
    max_num_decode_tokens: int,
    model_dtype: torch.dtype,
) -> None:
    """Compile TurboQuant decode kernels without running model forward.

    V1 dummy/profile warmup can avoid the TurboQuant decode path, which leaves
    `_tq_decode_stage1` and `_tq_decode_stage2` to compile on the first real
    decode request. This warmup calls the backend decode path with synthetic
    tensors whose launch-time constants match the runtime attention layer.
    """
    if max_num_decode_tokens <= 0:
        return

    seen: set[_TurboQuantDecodeWarmupKey] = set()
    num_warmups = 0

    for layer, impl in _iter_turboquant_attention_layers(model):
        key = _make_warmup_key(
            impl,
            block_size=block_size,
            block_table_stride=block_table_stride,
            model_dtype=model_dtype,
        )
        if key in seen:
            continue
        seen.add(key)
        _warmup_turboquant_decode_layer(
            layer,
            impl,
            device=device,
            block_size=block_size,
            block_table_stride=block_table_stride,
            max_num_decode_tokens=max_num_decode_tokens,
            model_dtype=model_dtype,
        )
        num_warmups += 1

    if num_warmups > 0:
        torch.accelerator.synchronize()
        logger.info("Warmed up %d TurboQuant decode kernel variant(s).", num_warmups)
