# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation

logger = init_logger(__name__)

if current_platform.is_cuda():
    try:
        import vllm._flashmla_C  # noqa: F401

        _flashmla_C_AVAILABLE = True
    except ImportError:
        _flashmla_C_AVAILABLE = False
else:
    _flashmla_C_AVAILABLE = False

if current_platform.is_cuda():
    try:
        import vllm._flashmla_extension_C  # noqa: F401

        _flashmla_extension_C_AVAILABLE = True
    except ImportError:
        _flashmla_extension_C_AVAILABLE = False
else:
    _flashmla_extension_C_AVAILABLE = False


def _is_flashmla_available() -> tuple[bool, str | None]:
    if not _flashmla_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_C is not available, likely was not "
            "compiled due to insufficient nvcc version or a supported arch "
            "was not in the list of target arches to compile for.",
        )
    if not _flashmla_extension_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_extension_C is not available, likely "
            "was not compiled due to a build error.",
        )

    return True, None


def is_flashmla_dense_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not current_platform.is_device_capability_family(90):
        return False, "FlashMLA Dense is only supported on Hopper devices."
    return True, None


def is_flashmla_sparse_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not (
        current_platform.is_device_capability_family(90)
        or current_platform.is_device_capability_family(100)
    ):
        return (
            False,
            "FlashMLA Sparse is only supported on Hopper and Blackwell DC devices.",
        )
    return True, None


def _raise_flashmla_unavailable(*_args, **_kwargs):
    _, reason = _is_flashmla_available()
    raise RuntimeError(reason or "FlashMLA is not available")


if _is_flashmla_available()[0]:
    from vllm.third_party.flashmla.flash_mla_interface import (  # noqa: F401
        FlashMLASchedMeta,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )
else:

    class FlashMLASchedMeta:  # type: ignore[no-redef]
        pass

    flash_attn_varlen_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_kvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_qkvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_mla_sparse_fwd = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_mla_with_kvcache = _raise_flashmla_unavailable  # type: ignore[assignment]
    get_mla_metadata = _raise_flashmla_unavailable  # type: ignore[assignment]


def get_mla_metadata_dense_fp8(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    return torch.ops._flashmla_extension_C.get_mla_decoding_metadata_dense_fp8(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
    )


def flash_mla_with_kvcache_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = torch.ops._flashmla_extension_C.fwd_kvcache_mla_fp8(
        q,
        k_cache,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        descale_q,
        descale_k,
    )
    return out, softmax_lse


# ---------------------------------------------------------------------------
# FlashMLA mega attention: Q norm + Q RoPE + sparse attention + inverse RoPE +
# FP8 cast of the output, in one kernel, for DeepSeek V4 / V4.1 on SM100.
# ---------------------------------------------------------------------------

MEGA_ATTN_WV_GROUP_SIZE = 8
_MEGA_ATTN_HEAD_DIM_V = 512
_MEGA_ATTN_ROPE_DIM = 64
# The kernel always emits per-32 scales, whatever num_per_channels says.
_MEGA_ATTN_QUANT_GROUP = 32
# Fixed kernel conventions: GPT-J (non-neox) RoPE, TMA-aligned col-major
# scales, round-to-nearest, packed ue8m0 -- i.e. what DeepGEMM's fp8_einsum
# consumes. Q norm is off: the V4.1 layer norms qr before wq_b.
_MEGA_ATTN_TAIL_ARGS = (False, _MEGA_ATTN_ROPE_DIM)
_MEGA_ATTN_SF_ARGS = (_MEGA_ATTN_QUANT_GROUP, True, True, True)


def is_flashmla_mega_attn_supported() -> tuple[bool, str | None]:
    """Whether FlashMLA's fused mega-attention kernels are usable here."""
    is_available, maybe_reason = is_flashmla_sparse_supported()
    if not is_available:
        return False, maybe_reason
    if not current_platform.is_device_capability_family(100):
        return False, "FlashMLA mega attention requires sm_10x GPUs."
    if not hasattr(torch.ops._flashmla_C, "fused_norm_rope_attn_rope_cast_decode"):
        return False, "vllm._flashmla_C was built without the mega attention ops."
    if not hasattr(torch.ops._flashmla_C, "permute_q_b_proj"):
        return False, "vllm._flashmla_C predates the mega attention weight permutes."
    return True, None


def alloc_mega_attn_output(
    num_tokens: int,
    n_wv_group: int,
    device: torch.device,
) -> "QuantizedActivation":
    """Allocate the output buffer pair the mega-attention kernels write into.

    One pair per forward step: the prefill and decode segments fill disjoint
    token ranges, so a single ``fp8_einsum`` over all ``num_tokens`` can
    consume the result instead of one call per segment.

    The scale buffer is MN-major -- stride 1 along tokens, head-dim stride
    ``ceil4(num_tokens)`` -- which is both what the kernel requires of a
    caller-provided buffer and what DeepGEMM expects for this N.
    """
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
    from vllm.utils.deep_gemm import get_tma_aligned_size

    d = MEGA_ATTN_WV_GROUP_SIZE * _MEGA_ATTN_HEAD_DIM_V
    data = torch.empty(
        (num_tokens, n_wv_group, d), dtype=torch.float8_e4m3fn, device=device
    )
    aligned = get_tma_aligned_size(num_tokens, torch.int32.itemsize)
    scale = torch.empty(
        (n_wv_group, d // (_MEGA_ATTN_QUANT_GROUP * 4), aligned),
        dtype=torch.int32,
        device=device,
    ).permute(2, 0, 1)[:num_tokens]
    return QuantizedActivation(
        data=data,
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=data.shape,
        quant_key=kMxfp8Dynamic,
    )


def mega_attn_token_range(
    out: "QuantizedActivation", start: int, end: int
) -> "QuantizedActivation":
    """The ``[start, end)`` token slice of a mega-attention output buffer."""
    from dataclasses import replace

    data = out.data[start:end]
    return replace(out, data=data, scale=out.scale[start:end], orig_shape=data.shape)


def flash_mla_mega_attn_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    token_positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_wv_group: int,
    out: "QuantizedActivation",
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mega attention over a non-paged bf16 ``kv``; writes ``out`` in place.

    Args:
        q: ``[s_q, h_q, 512]`` bf16 in the kernel's chunk-interleaved layout,
            before RoPE.
        kv: ``[s_kv, 1, 512]`` bf16, RoPE already applied.
        indices: ``[s_q, 1, topk]`` int32; entries outside ``[0, s_kv)`` skip.
        token_positions: ``[s_q]`` int32 RoPE positions of the queries.
        cos_sin_cache: ``[max_pos, 64]`` fp32, cos in ``[:, :32]``.
        n_wv_group: ``h_q // 8``.
        out: this segment's token slice of :func:`alloc_mega_attn_output`.

    Returns:
        ``(max_logits, lse)``, both ``[s_q, h_q]`` fp32.
    """
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    _, _, max_logits, lse = torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_fwd(
        q,
        kv,
        indices,
        sm_scale,
        _MEGA_ATTN_HEAD_DIM_V,
        attn_sink,
        topk_length,
        False,  # enable_q_norm
        0.0,  # rms_norm_eps
        token_positions,
        *_MEGA_ATTN_TAIL_ARGS,
        cos_sin_cache,
        n_wv_group,
        *_MEGA_ATTN_SF_ARGS,
        out.data,
        out.scale,
    )
    return max_logits, lse


def flash_mla_mega_attn_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    token_positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_wv_group: int,
    out: "QuantizedActivation",
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mega attention over paged quantized caches; writes ``out`` in place.

    ``k_cache`` / ``extra_k_cache`` are ``[num_blocks, page, 1, bytes]``, the
    format taken from ``bytes`` (584 V4, 528 V4.1 fp8, 288 V4.1 fp4; fp4 only
    as ``extra_k_cache`` beside a V4.1 fp8 ``k_cache``). ``indices`` are
    ``[s_q, topk]`` int32 slot ids (``block * page + offset``, ``-1`` invalid).

    Returns:
        ``lse``, ``[s_q, h_q]`` fp32.
    """
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    _, _, lse = torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_decode(
        q,
        k_cache,
        indices,
        sm_scale,
        _MEGA_ATTN_HEAD_DIM_V,
        attn_sink,
        topk_length,
        extra_k_cache,
        extra_indices,
        extra_topk_length,
        False,  # enable_q_norm
        0.0,  # rms_norm_eps
        token_positions,
        *_MEGA_ATTN_TAIL_ARGS,
        cos_sin_cache,
        n_wv_group,
        *_MEGA_ATTN_SF_ARGS,
        out.data,
        out.scale,
    )
    return lse
