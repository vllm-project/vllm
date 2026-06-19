# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Triton unified attention kernel specializations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from vllm.config import get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode, get_kv_quant_mode

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


@dataclass(frozen=True)
class TritonUnifiedAttentionWarmupKey:
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    q_dtype: torch.dtype
    cache_dtype: torch.dtype
    kv_quant_mode: KVQuantMode
    scale: float
    sliding_window: tuple[int, int]
    softcap: float
    use_alibi: bool
    use_alibi_sqrt: bool
    use_sinks: bool
    chunk_lookback: int
    use_td: bool


@dataclass(frozen=True)
class _WarmupShape:
    name: str
    query_lens: tuple[int, ...]
    seq_lens: tuple[int, ...]
    use_3d: bool


def _is_triton_attention_backend(backend: object) -> bool:
    try:
        return backend.get_name() == "TRITON_ATTN"  # type: ignore[attr-defined]
    except NotImplementedError:
        return False


def _iter_triton_attention_groups(
    runner: GPUModelRunner,
) -> Iterable[AttentionGroup]:
    for groups in runner.attn_groups:
        for group in groups:
            if _is_triton_attention_backend(group.backend):
                yield group


def _iter_triton_unified_attention_warmup_keys(
    runner: GPUModelRunner,
) -> list[TritonUnifiedAttentionWarmupKey]:
    keys: list[TritonUnifiedAttentionWarmupKey] = []
    seen: set[TritonUnifiedAttentionWarmupKey] = set()

    for group in _iter_triton_attention_groups(runner):
        # AttentionLayerBase is the runtime base for attention layers here.
        # Existing call sites use the same type-abstract escape hatch.
        layers = get_layers_from_vllm_config(
            runner.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
            group.layer_names,
        )
        for layer in layers.values():
            impl = getattr(layer, "impl", None)
            if impl is None or getattr(impl, "attn_type", AttentionType.DECODER) in (
                AttentionType.ENCODER,
                AttentionType.ENCODER_ONLY,
            ):
                continue

            kv_quant_mode = get_kv_quant_mode(impl.kv_cache_dtype)
            if kv_quant_mode.is_nvfp4:
                # Main does not route NVFP4 through TritonAttentionBackend yet.
                # The hook is intentionally generic so the NVFP4 branch can add
                # its own synthetic tensors without changing kernel_warmup.py.
                continue

            kv_cache_spec = cast(AttentionSpec, group.kv_cache_spec)
            key = TritonUnifiedAttentionWarmupKey(
                num_query_heads=impl.num_heads,
                num_kv_heads=impl.num_kv_heads,
                head_size=impl.head_size,
                block_size=group.kv_cache_spec.block_size,
                q_dtype=runner.model_config.dtype,
                cache_dtype=kv_cache_spec.dtype,
                kv_quant_mode=kv_quant_mode,
                scale=impl.scale,
                sliding_window=impl.sliding_window,
                softcap=impl.logits_soft_cap,
                use_alibi=impl.alibi_slopes is not None,
                use_alibi_sqrt=impl.use_alibi_sqrt,
                use_sinks=impl.sinks is not None,
                chunk_lookback=impl.chunk_lookback,
                use_td=impl.use_td,
            )
            if key not in seen:
                seen.add(key)
                keys.append(key)

    return keys


def _default_seq_threshold_3d(key: TritonUnifiedAttentionWarmupKey) -> int:
    return max(1, 128 // max(key.num_kv_heads, 1))


def _seq_threshold_3d(
    key: TritonUnifiedAttentionWarmupKey,
    cudagraph_batch_sizes: tuple[int, ...],
) -> int:
    threshold = _default_seq_threshold_3d(key)
    if not cudagraph_batch_sizes:
        return threshold

    # Match TritonAttentionMetadataBuilder: when decode CUDA graphs are
    # enabled, the 3D/2D cutoff is the capture size closest to the default
    # threshold so each captured graph stays on a stable kernel path.
    return min(cudagraph_batch_sizes, key=lambda size: abs(size - threshold))


def _decode_warmup_batch_sizes(
    key: TritonUnifiedAttentionWarmupKey,
    cudagraph_batch_sizes: tuple[int, ...],
) -> tuple[int, ...]:
    seq_threshold_3d = _seq_threshold_3d(key, cudagraph_batch_sizes)
    if not cudagraph_batch_sizes:
        return (max(1, min(4, seq_threshold_3d)), seq_threshold_3d + 1)

    selected = {size for size in cudagraph_batch_sizes if size <= seq_threshold_3d}
    larger_sizes = [size for size in cudagraph_batch_sizes if size > seq_threshold_3d]
    if larger_sizes:
        selected.add(larger_sizes[0])
    else:
        selected.add(seq_threshold_3d + 1)
    return tuple(sorted(selected))


def _warmup_shapes(
    key: TritonUnifiedAttentionWarmupKey,
    cudagraph_batch_sizes: tuple[int, ...] = (),
) -> tuple[_WarmupShape, ...]:
    seq_threshold_3d = _seq_threshold_3d(key, cudagraph_batch_sizes)
    decode_batch_sizes = _decode_warmup_batch_sizes(key, cudagraph_batch_sizes)
    decode_kv_len = key.block_size * 2
    prefill_len = min(16, key.block_size)

    return (
        _WarmupShape("prefill_2d", (prefill_len,), (prefill_len,), False),
        *(
            _WarmupShape(
                "decode_3d" if size <= seq_threshold_3d else "decode_2d",
                (1,) * size,
                (decode_kv_len,) * size,
                size <= seq_threshold_3d,
            )
            for size in decode_batch_sizes
        ),
    )


def _kv_cache_dtype(key: TritonUnifiedAttentionWarmupKey) -> torch.dtype:
    if key.kv_quant_mode == KVQuantMode.FP8_PER_TENSOR:
        return current_platform.fp8_dtype()
    if key.kv_quant_mode == KVQuantMode.FP8_PER_TOKEN_HEAD:
        return current_platform.fp8_dtype()
    if key.kv_quant_mode == KVQuantMode.INT8_PER_TOKEN_HEAD:
        return torch.int8
    return key.cache_dtype


def _allocate_kv_cache_tensors(
    key: TritonUnifiedAttentionWarmupKey,
    shape: _WarmupShape,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    max_seq_len = max(shape.seq_lens)
    num_blocks = (max_seq_len + key.block_size - 1) // key.block_size
    cache_dtype = _kv_cache_dtype(key)

    head_size = key.head_size
    if key.kv_quant_mode.is_per_token_head:
        scale_pad = get_dtype_size(torch.float32) // get_dtype_size(cache_dtype)
        head_size += scale_pad

    k = torch.empty(
        (num_blocks, key.block_size, key.num_kv_heads, head_size),
        dtype=cache_dtype,
        device=device,
    )
    v = torch.empty_like(k)

    k_scale_cache = None
    v_scale_cache = None
    if key.kv_quant_mode.is_per_token_head:
        k_scale_cache = torch.ones(
            (num_blocks, key.block_size, key.num_kv_heads),
            dtype=torch.float32,
            device=device,
        )
        v_scale_cache = torch.ones_like(k_scale_cache)

    return k, v, k_scale_cache, v_scale_cache


def _make_cu_seqlens(query_lens: tuple[int, ...], device: torch.device) -> torch.Tensor:
    prefix_sums = torch.cumsum(torch.tensor(query_lens), 0).tolist()
    return torch.tensor((0, *prefix_sums), dtype=torch.int32, device=device)


def _warmup_unified_attention_key(
    key: TritonUnifiedAttentionWarmupKey,
    device: torch.device,
    cudagraph_batch_sizes: tuple[int, ...] = (),
) -> None:
    from vllm.v1.attention.backends.triton_attn import NUM_PAR_SOFTMAX_SEGMENTS
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    for shape in _warmup_shapes(key, cudagraph_batch_sizes):
        num_tokens = sum(shape.query_lens)
        max_q_len = max(shape.query_lens)
        max_seq_len = max(shape.seq_lens)
        num_seqs = len(shape.query_lens)

        q = torch.empty(
            (num_tokens, key.num_query_heads, key.head_size),
            dtype=key.q_dtype,
            device=device,
        )
        out = torch.empty_like(q)
        k, v, k_scale_cache, v_scale_cache = _allocate_kv_cache_tensors(
            key, shape, device
        )
        cu_seqlens_q = _make_cu_seqlens(shape.query_lens, device)
        seq_lens = torch.tensor(shape.seq_lens, dtype=torch.int32, device=device)
        max_blocks = (max_seq_len + key.block_size - 1) // key.block_size
        block_table = (
            torch.arange(max_blocks, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .expand(num_seqs, max_blocks)
            .contiguous()
        )

        descale_shape = (num_seqs, key.num_kv_heads)
        k_descale = None
        v_descale = None
        if key.kv_quant_mode == KVQuantMode.FP8_PER_TENSOR:
            k_descale = torch.ones(descale_shape, dtype=torch.float32, device=device)
            v_descale = torch.ones(descale_shape, dtype=torch.float32, device=device)

        alibi_slopes = None
        if key.use_alibi:
            alibi_slopes = torch.zeros(
                key.num_query_heads, dtype=torch.float32, device=device
            )

        sinks = None
        if key.use_sinks:
            sinks = torch.zeros(key.num_query_heads, dtype=torch.float32, device=device)

        seq_threshold_3d = None
        num_par_softmax_segments = None
        softmax_segm_output = None
        softmax_segm_max = None
        softmax_segm_expsum = None
        if shape.use_3d:
            seq_threshold_3d = num_seqs
            num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
            head_size_padded = 1 << (key.head_size - 1).bit_length()
            softmax_segm_output = torch.empty(
                (
                    seq_threshold_3d,
                    key.num_query_heads,
                    num_par_softmax_segments,
                    head_size_padded,
                ),
                dtype=torch.float32,
                device=device,
            )
            softmax_segm_max = torch.empty(
                (seq_threshold_3d, key.num_query_heads, num_par_softmax_segments),
                dtype=torch.float32,
                device=device,
            )
            softmax_segm_expsum = torch.empty_like(softmax_segm_max)

        unified_attention(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_q_len,
            seqused_k=seq_lens,
            max_seqlen_k=max_seq_len,
            softmax_scale=key.scale,
            causal=True,
            window_size=key.sliding_window,
            block_table=block_table,
            softcap=key.softcap,
            q_descale=None,
            k_descale=k_descale,
            v_descale=v_descale,
            seq_threshold_3D=seq_threshold_3d,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            alibi_slopes=alibi_slopes,
            sinks=sinks,
            use_alibi_sqrt=key.use_alibi_sqrt,
            kv_quant_mode=key.kv_quant_mode,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            chunk_lookback=key.chunk_lookback,
            use_td=key.use_td,
        )


def triton_unified_attention_warmup(runner: GPUModelRunner) -> None:
    if runner.is_pooling_model or not runner.attn_groups:
        return
    if not (current_platform.is_cuda() or current_platform.is_xpu()):
        return

    keys = _iter_triton_unified_attention_warmup_keys(runner)
    if not keys:
        return

    logger.info(
        "Warming up Triton unified attention kernels for %d configuration(s).",
        len(keys),
    )
    cudagraph_batch_sizes = tuple(getattr(runner, "cudagraph_batch_sizes", ()))
    for key in keys:
        try:
            _warmup_unified_attention_key(
                key, runner.device, cudagraph_batch_sizes=cudagraph_batch_sizes
            )
        except Exception:
            logger.warning(
                "Triton unified attention warmup failed for %s.", key, exc_info=True
            )
