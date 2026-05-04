# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for DeepGEMM API changes.

Users of vLLM should always import **only** these wrappers.
"""

import functools
import importlib
import os
from collections.abc import Callable
from enum import Enum
from typing import Any, NoReturn

import torch

import vllm.envs as envs
from vllm.logger import logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.math_utils import cdiv

_DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES: set[str] = {
    "qwen3_5_text",
    "qwen3_5_moe_text",
}


def should_auto_disable_deep_gemm(model_type: str | None) -> bool:
    """Check if DeepGemm should be auto-disabled for this model on Blackwell.

    Returns True if the model is known to have accuracy degradation with
    DeepGemm's E8M0 scale format on Blackwell GPUs (SM100+).
    """
    if model_type is None:
        return False
    if not current_platform.is_device_capability_family(100):
        return False
    return model_type in _DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES


class DeepGemmQuantScaleFMT(Enum):
    # Float32 scales in Float32 tensor
    FLOAT32 = 0
    # Compute float32 scales and ceil the scales to UE8M0.
    # Keep the scales in Float32 tensor.
    FLOAT32_CEIL_UE8M0 = 1
    # Compute float32 scales and ceil the scales to UE8M0.
    # Pack the scales into a int32 tensor where each int32
    # element contains 4 scale values.
    UE8M0 = 2

    @classmethod
    def init_oracle_cache(cls) -> None:
        """Initialize the oracle decision and store it in the class cache"""
        cached = getattr(cls, "_oracle_cache", None)
        if cached is not None:
            return

        use_e8m0 = (
            envs.VLLM_USE_DEEP_GEMM_E8M0
            and is_deep_gemm_supported()
            and (_fp8_gemm_nt_impl is not None)
        )
        if not use_e8m0:
            cls._oracle_cache = cls.FLOAT32  # type: ignore
            return

        cls._oracle_cache = (  # type: ignore
            cls.UE8M0
            if current_platform.is_device_capability_family(100)
            else cls.FLOAT32_CEIL_UE8M0
        )

    @classmethod
    def from_oracle(cls) -> "DeepGemmQuantScaleFMT":
        """Return the pre-initialized oracle decision"""
        cached = getattr(cls, "_oracle_cache", None)
        assert cached is not None, "DeepGemmQuantScaleFMT oracle cache not initialized"
        return cached


@functools.cache
def is_deep_gemm_supported() -> bool:
    """Return `True` if DeepGEMM is supported on the current platform.
    Currently, only Hopper and Blackwell GPUs are supported.
    """
    is_supported_arch = current_platform.support_deep_gemm()
    return envs.VLLM_USE_DEEP_GEMM and has_deep_gemm() and is_supported_arch


@functools.cache
def is_deep_gemm_e8m0_used() -> bool:
    """Return `True` if vLLM is configured to use DeepGEMM "
    "E8M0 scale on a Hopper or Blackwell-class GPU.
    """
    if not is_deep_gemm_supported():
        logger.debug_once(
            "DeepGEMM E8M0 disabled: DeepGEMM not supported on this system."
        )
        return False

    _lazy_init()

    if _fp8_gemm_nt_impl is None:
        logger.info_once("DeepGEMM E8M0 disabled: _fp8_gemm_nt_impl not found")
        return False

    if envs.VLLM_USE_DEEP_GEMM_E8M0:
        logger.info_once("DeepGEMM E8M0 enabled on current platform.")
        return True

    logger.info_once("DeepGEMM E8M0 disabled on current configuration.")
    return False


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable DeepGEMM backend."""
    raise RuntimeError(
        "DeepGEMM backend is not available or outdated. Please install or "
        "update the `deep_gemm` to a newer version to enable FP8 kernels."
    )


_cublaslt_gemm_nt_impl: Callable[..., Any] | None = None
_fp8_gemm_nt_impl: Callable[..., Any] | None = None
_fp8_einsum_impl: Callable[..., Any] | None = None
_grouped_impl: Callable[..., Any] | None = None
_grouped_masked_impl: Callable[..., Any] | None = None
_grouped_fp4_impl: Callable[..., Any] | None = None
_fp8_fp4_mqa_logits_impl: Callable[..., Any] | None = None
_fp8_fp4_paged_mqa_logits_impl: Callable[..., Any] | None = None
_get_paged_mqa_logits_metadata_impl: Callable[..., Any] | None = None
_tf32_hc_prenorm_gemm_impl: Callable[..., Any] | None = None
_get_mn_major_tma_aligned_tensor_impl: Callable[..., Any] | None = None
_get_mk_alignment_for_contiguous_layout_impl: Callable[..., Any] | None = None
_transform_sf_into_required_layout_impl: Callable[..., Any] | None = None


def _import_deep_gemm():
    """Import the deep_gemm module.

    Prefers an externally installed ``deep_gemm`` package (so users can
    pin a specific version), then falls back to the vendored copy bundled
    in the vLLM wheel.

    Returns ``None`` when neither source is usable.
    """
    # 1. Try the external (pip-installed) package first.
    try:
        module = importlib.import_module("deep_gemm")
        logger.debug_once("Imported deep_gemm module from site-packages")
        return module
    except ImportError:
        logger.debug_once(
            "deep_gemm not found in site-packages, "
            "trying vendored vllm.third_party.deep_gemm"
        )

    # 2. Fall back to the vendored copy bundled in the vLLM wheel.
    try:
        module = importlib.import_module("vllm.third_party.deep_gemm")
        logger.debug_once("Imported deep_gemm module from vllm.third_party.deep_gemm")
        return module
    except ImportError:
        logger.debug_once("Vendored deep_gemm not found either")
    except Exception as e:
        # The vendored module may raise RuntimeError during _C.init()
        # if JIT include files are missing (e.g. incomplete wheel).
        logger.warning_once("Failed to import vendored deep_gemm: %s", e)

    return None


def _lazy_init() -> None:
    """Import deep_gemm and resolve symbols on first use."""
    global _cublaslt_gemm_nt_impl
    global _fp8_gemm_nt_impl, _fp8_einsum_impl
    global _grouped_impl, _grouped_masked_impl, _grouped_fp4_impl
    global _fp8_fp4_mqa_logits_impl, _fp8_fp4_paged_mqa_logits_impl
    global _get_paged_mqa_logits_metadata_impl
    global _tf32_hc_prenorm_gemm_impl
    global _get_mn_major_tma_aligned_tensor_impl
    global _get_mk_alignment_for_contiguous_layout_impl
    global _transform_sf_into_required_layout_impl
    # fast path
    if (
        _cublaslt_gemm_nt_impl is not None
        or _fp8_gemm_nt_impl is not None
        or _fp8_einsum_impl is not None
        or _grouped_impl is not None
        or _grouped_masked_impl is not None
        or _grouped_fp4_impl is not None
        or _fp8_fp4_mqa_logits_impl is not None
        or _fp8_fp4_paged_mqa_logits_impl is not None
        or _get_paged_mqa_logits_metadata_impl is not None
        or _tf32_hc_prenorm_gemm_impl is not None
        or _get_mk_alignment_for_contiguous_layout_impl is not None
        or _transform_sf_into_required_layout_impl is not None
    ):
        return

    if not has_deep_gemm():
        return

    # Set up deep_gemm cache path
    DEEP_GEMM_JIT_CACHE_ENV_NAME = "DG_JIT_CACHE_DIR"
    if not os.environ.get(DEEP_GEMM_JIT_CACHE_ENV_NAME, None):
        os.environ[DEEP_GEMM_JIT_CACHE_ENV_NAME] = os.path.join(
            envs.VLLM_CACHE_ROOT, "deep_gemm"
        )

    _dg = _import_deep_gemm()
    if _dg is None:
        return

    _cublaslt_gemm_nt_impl = getattr(_dg, "cublaslt_gemm_nt", None)
    _fp8_gemm_nt_impl = getattr(_dg, "fp8_gemm_nt", None)
    _fp8_einsum_impl = getattr(_dg, "fp8_einsum", None)
    _grouped_impl = getattr(_dg, "m_grouped_fp8_gemm_nt_contiguous", None)
    _grouped_masked_impl = getattr(_dg, "fp8_m_grouped_gemm_nt_masked", None)
    _grouped_fp4_impl = getattr(_dg, "m_grouped_fp8_fp4_gemm_nt_contiguous", None)
    # DeepGEMM exposes fp8_fp4_*_mqa_logits as the canonical symbols that
    # handle both the FP8 and FP4 Q/K paths via a tuple-typed `q`.
    _fp8_fp4_mqa_logits_impl = getattr(_dg, "fp8_fp4_mqa_logits", None)
    _fp8_fp4_paged_mqa_logits_impl = getattr(_dg, "fp8_fp4_paged_mqa_logits", None)
    _get_paged_mqa_logits_metadata_impl = getattr(
        _dg, "get_paged_mqa_logits_metadata", None
    )
    _tf32_hc_prenorm_gemm_impl = getattr(_dg, "tf32_hc_prenorm_gemm", None)
    _get_mn_major_tma_aligned_tensor_impl = getattr(
        _dg, "get_mn_major_tma_aligned_tensor", None
    )
    _get_mk_alignment_for_contiguous_layout_impl = getattr(
        _dg, "get_mk_alignment_for_contiguous_layout", None
    )
    _transform_sf_into_required_layout_impl = getattr(
        _dg, "transform_sf_into_required_layout", None
    )
    DeepGemmQuantScaleFMT.init_oracle_cache()


def get_num_sms() -> int:
    _lazy_init()
    dg = _import_deep_gemm()
    if dg is None:
        raise RuntimeError("DeepGEMM is not available")
    return int(dg.get_num_sms())


def set_num_sms(num_sms: int) -> None:
    _lazy_init()
    dg = _import_deep_gemm()
    if dg is None:
        raise RuntimeError("DeepGEMM is not available")
    dg.set_num_sms(num_sms)


@functools.cache
def get_mk_alignment_for_contiguous_layout() -> list[int]:
    _lazy_init()
    if _get_mk_alignment_for_contiguous_layout_impl is None:
        return _missing()
    mk_align_size = _get_mk_alignment_for_contiguous_layout_impl()
    return [mk_align_size, mk_align_size]


def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for DeepGEMM's get_mn_major_tma_aligned_tensor"""
    _lazy_init()
    if _get_mn_major_tma_aligned_tensor_impl is None:
        return _missing()
    return _get_mn_major_tma_aligned_tensor_impl(x)


def cublaslt_gemm_nt(*args, **kwargs):
    _lazy_init()
    if _cublaslt_gemm_nt_impl is None:
        return _missing(*args, **kwargs)
    return _cublaslt_gemm_nt_impl(*args, **kwargs)


def fp8_gemm_nt(*args, **kwargs):
    _lazy_init()
    if _fp8_gemm_nt_impl is None:
        return _missing(*args, **kwargs)
    if "is_deep_gemm_e8m0_used" in kwargs:
        use_ue8m0 = kwargs["is_deep_gemm_e8m0_used"]
        del kwargs["is_deep_gemm_e8m0_used"]
    else:
        use_ue8m0 = is_deep_gemm_e8m0_used()
    return _fp8_gemm_nt_impl(*args, disable_ue8m0_cast=not use_ue8m0, **kwargs)


def fp8_einsum(*args, **kwargs):
    _lazy_init()
    if _fp8_einsum_impl is None:
        return _missing(*args, **kwargs)
    return _fp8_einsum_impl(*args, **kwargs)


def m_grouped_fp8_gemm_nt_contiguous(*args, **kwargs):
    _lazy_init()
    if _grouped_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def m_grouped_fp8_fp4_gemm_nt_contiguous(*args, **kwargs):
    _lazy_init()
    if _grouped_fp4_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_fp4_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def fp8_m_grouped_gemm_nt_masked(*args, **kwargs):
    _lazy_init()
    if _grouped_masked_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_masked_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def transform_sf_into_required_layout(*args, **kwargs):
    _lazy_init()
    if _transform_sf_into_required_layout_impl is None:
        return _missing(*args, **kwargs)
    return _transform_sf_into_required_layout_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


_SM120_MQA_LOGITS_MAX_SCORE_BYTES = 64 * 1024 * 1024
_SM120_PAGED_MQA_TOPK_CHUNK_SIZE = 8192


def _fp8_mqa_logits_head_chunk_size(
    seq_len: int,
    seq_len_kv: int,
    num_heads: int,
) -> int:
    # The SM120 torch path is used on long prefill paths where materializing
    # [head_chunk, M, N] scores can otherwise allocate multiple GiB. Keep the
    # transient score tensor bounded, while still using larger head chunks for
    # short prompts where they are faster.
    score_elems_per_head = max(1, seq_len * seq_len_kv)
    max_heads = _SM120_MQA_LOGITS_MAX_SCORE_BYTES // (score_elems_per_head * 4)
    return max(1, min(8, num_heads, max_heads))


def _fp8_mqa_logits_k_chunk_size(
    seq_len: int,
    seq_len_kv: int,
    head_chunk_size: int,
) -> int:
    score_elems_per_key = max(1, seq_len * head_chunk_size)
    max_keys = _SM120_MQA_LOGITS_MAX_SCORE_BYTES // (score_elems_per_key * 4)
    return max(1, min(seq_len_kv, max_keys))


def _fp8_mqa_logits_torch(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("SM120 MQA logits torch path only supports FP8 Q")

    k_values, k_scales = kv
    k_f32 = k_values.to(torch.float32)
    k_f32.mul_(k_scales.reshape(-1, 1).to(torch.float32))
    k_t = k_f32.transpose(0, 1).contiguous()

    seq_len, num_heads, _ = q_values.shape
    seq_len_kv = k_f32.shape[0]
    logits = torch.zeros(
        (seq_len, seq_len_kv), device=q_values.device, dtype=torch.float32
    )
    head_chunk_size = _fp8_mqa_logits_head_chunk_size(seq_len, seq_len_kv, num_heads)

    for head_start in range(0, num_heads, head_chunk_size):
        head_end = min(head_start + head_chunk_size, num_heads)
        q_chunk = q_values[:, head_start:head_end, :].to(torch.float32)
        q_chunk = q_chunk.transpose(0, 1).contiguous()
        head_weights = weights[:, head_start:head_end].transpose(0, 1).unsqueeze(-1)
        k_chunk_size = _fp8_mqa_logits_k_chunk_size(
            seq_len, seq_len_kv, head_end - head_start
        )
        for k_start in range(0, seq_len_kv, k_chunk_size):
            k_end = min(k_start + k_chunk_size, seq_len_kv)
            scores = torch.matmul(q_chunk, k_t[:, k_start:k_end])
            scores.relu_()
            scores.mul_(head_weights)
            logits[:, k_start:k_end].add_(
                scores[0] if scores.shape[0] == 1 else scores.sum(dim=0)
            )

    if clean_logits:
        offsets = torch.arange(seq_len_kv, device=q_values.device)
        valid = (offsets[None, :] >= cu_seqlen_ks[:, None]) & (
            offsets[None, :] < cu_seqlen_ke[:, None]
        )
        logits = logits.masked_fill(~valid, float("-inf"))

    return logits


def _fp8_mqa_logits_topk_torch(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_tokens: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("SM120 MQA top-k torch path only supports FP8 Q")

    k_values, k_scales = kv
    k_f32 = k_values.to(torch.float32)
    k_f32.mul_(k_scales.reshape(-1, 1).to(torch.float32))
    k_t = k_f32.transpose(0, 1).contiguous()

    seq_len, num_heads, _ = q_values.shape
    seq_len_kv = k_f32.shape[0]
    if out is None:
        out = torch.empty(
            (seq_len, topk_tokens), device=q_values.device, dtype=torch.int32
        )
    else:
        assert out.shape == (seq_len, topk_tokens)
        assert out.dtype == torch.int32
    out.fill_(-1)

    best_values = torch.full(
        (seq_len, topk_tokens),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )
    head_chunk_size = _fp8_mqa_logits_head_chunk_size(seq_len, seq_len_kv, num_heads)
    k_chunk_size = _fp8_mqa_logits_k_chunk_size(seq_len, seq_len_kv, head_chunk_size)
    max_chunk_topk = min(topk_tokens, k_chunk_size)
    chunk_values_buf = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_indices_buf = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int64,
    )
    chunk_indices_i32 = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    candidate_values = torch.empty(
        (seq_len, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    candidate_indices = torch.empty(
        (seq_len, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    next_best_values = torch.empty_like(best_values)
    selected = torch.empty(
        (seq_len, topk_tokens),
        device=q_values.device,
        dtype=torch.int64,
    )

    for k_start in range(0, seq_len_kv, k_chunk_size):
        k_end = min(k_start + k_chunk_size, seq_len_kv)
        chunk_logits = torch.zeros(
            (seq_len, k_end - k_start),
            device=q_values.device,
            dtype=torch.float32,
        )
        for head_start in range(0, num_heads, head_chunk_size):
            head_end = min(head_start + head_chunk_size, num_heads)
            q_chunk = q_values[:, head_start:head_end, :].to(torch.float32)
            q_chunk = q_chunk.transpose(0, 1).contiguous()
            head_weights = weights[:, head_start:head_end].transpose(0, 1).unsqueeze(-1)
            scores = torch.matmul(q_chunk, k_t[:, k_start:k_end])
            scores.relu_()
            scores.mul_(head_weights)
            chunk_logits.add_(scores[0] if scores.shape[0] == 1 else scores.sum(dim=0))

        offsets = torch.arange(k_start, k_end, device=q_values.device)
        valid = (offsets[None, :] >= cu_seqlen_ks[:, None]) & (
            offsets[None, :] < cu_seqlen_ke[:, None]
        )
        chunk_logits.masked_fill_(~valid, float("-inf"))

        chunk_topk = min(topk_tokens, k_end - k_start)
        chunk_values = chunk_values_buf[:, :chunk_topk]
        chunk_indices = chunk_indices_buf[:, :chunk_topk]
        torch.topk(chunk_logits, chunk_topk, dim=1, out=(chunk_values, chunk_indices))
        chunk_indices_out = chunk_indices_i32[:, :chunk_topk]
        chunk_indices_out.copy_(chunk_indices)
        chunk_indices_out.add_(k_start)

        candidate_cols = topk_tokens + chunk_topk
        candidate_values_view = candidate_values[:, :candidate_cols]
        candidate_indices_view = candidate_indices[:, :candidate_cols]
        candidate_values_view[:, :topk_tokens].copy_(best_values)
        candidate_values_view[:, topk_tokens:candidate_cols].copy_(chunk_values)
        candidate_indices_view[:, :topk_tokens].copy_(out)
        candidate_indices_view[:, topk_tokens:candidate_cols].copy_(chunk_indices_out)
        torch.topk(
            candidate_values_view,
            topk_tokens,
            dim=1,
            out=(next_best_values, selected),
        )
        torch.gather(candidate_indices_view, 1, selected, out=out)
        best_values, next_best_values = next_best_values, best_values
        out.masked_fill_(~torch.isfinite(best_values), -1)

    return out


def fp8_fp4_mqa_topk_indices(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
) -> bool:
    """Write SM120 FP8 MQA top-k indices without materializing full logits."""
    _lazy_init()
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and q[1] is None
    ):
        return False
    _fp8_mqa_logits_topk_torch(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices.shape[1],
        out=topk_indices,
    )
    return True


def _fp8_mqa_logits_sm12x(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    q_values, q_scale = q
    if clean_logits and q_scale is None and q_values.dim() == 3 and kv[0].dim() == 2:
        from vllm.model_executor.layers.deepseek_v4_triton_kernels import (
            fp8_mqa_logits_triton,
        )

        return fp8_mqa_logits_triton(q_values, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
    return _fp8_mqa_logits_torch(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
    )


def fp8_fp4_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute MQA logits for a single sequence without KV paging.

    Unified FP8/FP4 dispatch — the underlying DeepGEMM kernel takes
    ``q = (values, scales_or_None)`` where ``scales`` is None for FP8 Q
    (per-token scale is folded into ``weights``) and a packed block-scale
    tensor for MXFP4 Q.

    Args:
        q: Tuple ``(q_values, q_scale)``. FP8 path: q_values is [M, H, D]
            float8_e4m3fn and q_scale is None (per-token scale is folded
            into ``weights``). FP4 path: q_values is packed uint8 and
            q_scale is the companion block-scale tensor.
        kv: Tuple `(k_packed, k_scales)` — FP8 layout is [N, D]
            float8_e4m3fn plus fp32 scales [N]; FP4 layout is packed uint8.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query
            position, shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query
            position, shape [M], dtype int32.
        clean_logits: Whether to clean the unfilled logits into `-inf`.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    _lazy_init()
    if current_platform.is_device_capability_family(120) and q[1] is None:
        return _fp8_mqa_logits_sm12x(
            q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
        )
    if _fp8_fp4_mqa_logits_impl is None:
        return _missing()
    return _fp8_fp4_mqa_logits_impl(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=clean_logits,
    )


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor, block_size: int, num_sms: int
) -> torch.Tensor:
    """Build scheduling metadata for paged MQA logits.

    Args:
        context_lens: Tensor of shape [B], dtype int32; effective context length
            per batch element.
        block_size: KV-cache block size in tokens (e.g., 64).
        num_sms: Number of SMs available. 132 for Hopper

    Returns:
        Backend-specific tensor consumed by `fp8_fp4_paged_mqa_logits` to
        schedule work across SMs.
    """
    _lazy_init()
    if _get_paged_mqa_logits_metadata_impl is None:
        return _missing()
    return _get_paged_mqa_logits_metadata_impl(context_lens, block_size, num_sms)


def _fp8_paged_mqa_logits_torch(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("SM120 paged MQA torch path only supports FP8 Q")

    batch_size, next_n, num_heads, head_dim = q_values.shape
    head_dim_with_scale = kv_cache.shape[-1]
    assert head_dim_with_scale > head_dim
    assert weights.shape == (batch_size * next_n, num_heads)
    assert context_lens.shape == (batch_size, next_n)

    from vllm.model_executor.layers.deepseek_v4_triton_kernels import (
        _view_packed_fp8_paged_mqa_kv_cache,
    )

    kv_values, kv_scales = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_kv, _, _ = kv_values.shape
    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )

    q_f32 = q_values.float()
    score_bytes = _SM120_MQA_LOGITS_MAX_SCORE_BYTES
    max_tokens_per_chunk = max(1, score_bytes // max(1, num_heads * 4))
    token_offsets_cache: dict[int, torch.Tensor] = {}

    for batch_idx in range(batch_size):
        for next_idx in range(next_n):
            row = batch_idx * next_n + next_idx
            context_len = int(context_lens[batch_idx, next_idx].item())
            if context_len <= 0:
                continue

            q_row = q_f32[batch_idx, next_idx]
            row_weights = weights[row]
            for token_start in range(0, context_len, max_tokens_per_chunk):
                token_end = min(context_len, token_start + max_tokens_per_chunk)
                chunk_len = token_end - token_start
                token_offsets = token_offsets_cache.get(chunk_len)
                if token_offsets is None or token_offsets.device != q_values.device:
                    token_offsets = torch.arange(
                        chunk_len, device=q_values.device, dtype=torch.long
                    )
                    token_offsets_cache[chunk_len] = token_offsets
                token_ids = token_start + token_offsets
                logical_blocks = token_ids // block_kv
                token_in_block = token_ids - logical_blocks * block_kv
                physical_blocks = block_tables[batch_idx, logical_blocks]
                kv_chunk = kv_values[physical_blocks, token_in_block, 0].float()
                scale_chunk = kv_scales[physical_blocks, token_in_block, 0].squeeze(-1)
                kv_chunk.mul_(scale_chunk[:, None])
                scores = torch.matmul(q_row, kv_chunk.T)
                scores.relu_()
                scores.mul_(row_weights[:, None])
                logits[row, token_start:token_end] = scores.sum(dim=0)

    return logits


def _fp8_paged_mqa_logits_sm12x(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    q_values, q_scale = q
    if (
        q_scale is None
        and q_values.dim() == 4
        and kv_cache.dtype == torch.uint8
        and kv_cache.shape[-1] == q_values.shape[-1] + 4
    ):
        from vllm.model_executor.layers.deepseek_v4_triton_kernels import (
            fp8_paged_mqa_logits_triton,
        )

        return fp8_paged_mqa_logits_triton(
            q_values, kv_cache, weights, context_lens, block_tables, max_model_len
        )
    return _fp8_paged_mqa_logits_torch(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )


def fp8_fp4_paged_mqa_topk_indices(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    topk_indices: torch.Tensor,
) -> bool:
    """Write SM120 FP8 paged MQA top-k indices without full logits."""
    _lazy_init()
    q_values, q_scale = q
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and q_scale is None
        and q_values.dim() == 4
        and kv_cache.dtype == torch.uint8
        and kv_cache.shape[-1] == q_values.shape[-1] + 4
    ):
        return False

    num_rows = q_values.shape[0] * q_values.shape[1]
    topk_tokens = topk_indices.shape[1]
    assert topk_indices.shape == (num_rows, topk_tokens)
    assert topk_indices.dtype == torch.int32
    topk_indices.fill_(-1)
    if num_rows == 0 or topk_tokens == 0 or max_model_len == 0:
        return True

    best_values = torch.full(
        (num_rows, topk_tokens),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_size = max(1, _SM120_PAGED_MQA_TOPK_CHUNK_SIZE)
    max_chunk_topk = min(topk_tokens, chunk_size)
    chunk_values_buf = torch.empty(
        (num_rows, max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_indices_buf = torch.empty(
        (num_rows, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int64,
    )
    chunk_indices_i32 = torch.empty(
        (num_rows, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    candidate_values = torch.empty(
        (num_rows, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    candidate_indices = torch.empty(
        (num_rows, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    next_best_values = torch.empty_like(best_values)
    selected = torch.empty(
        (num_rows, topk_tokens),
        device=q_values.device,
        dtype=torch.int64,
    )

    from vllm.model_executor.layers.deepseek_v4_triton_kernels import (
        fp8_paged_mqa_logits_triton,
    )

    for token_start in range(0, max_model_len, chunk_size):
        token_count = min(chunk_size, max_model_len - token_start)
        chunk_logits = fp8_paged_mqa_logits_triton(
            q_values,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len,
            token_start=token_start,
            token_count=token_count,
        )
        chunk_topk = min(topk_tokens, token_count)
        chunk_values = chunk_values_buf[:, :chunk_topk]
        chunk_indices = chunk_indices_buf[:, :chunk_topk]
        torch.topk(chunk_logits, chunk_topk, dim=1, out=(chunk_values, chunk_indices))
        chunk_indices_out = chunk_indices_i32[:, :chunk_topk]
        chunk_indices_out.copy_(chunk_indices)
        chunk_indices_out.add_(token_start)

        candidate_cols = topk_tokens + chunk_topk
        candidate_values_view = candidate_values[:, :candidate_cols]
        candidate_indices_view = candidate_indices[:, :candidate_cols]
        candidate_values_view[:, :topk_tokens].copy_(best_values)
        candidate_values_view[:, topk_tokens:candidate_cols].copy_(chunk_values)
        candidate_indices_view[:, :topk_tokens].copy_(topk_indices)
        candidate_indices_view[:, topk_tokens:candidate_cols].copy_(chunk_indices_out)
        torch.topk(
            candidate_values_view,
            topk_tokens,
            dim=1,
            out=(next_best_values, selected),
        )
        torch.gather(candidate_indices_view, 1, selected, out=topk_indices)
        best_values, next_best_values = next_best_values, best_values
        topk_indices.masked_fill_(~torch.isfinite(best_values), -1)

    return True


def fp8_fp4_paged_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute MQA logits using a paged KV-cache.

    Unified FP8/FP4 dispatch — the underlying DeepGEMM kernel takes
    ``q = (values, scales_or_None)``; pass ``(q_tensor, None)`` for the FP8
    path and ``(q_values, q_scale)`` for MXFP4.

    Args:
        q: Tuple ``(q_values, q_scale)``. FP8 path: q_values is
            [B, next_n, H, D] float8_e4m3fn and q_scale is None. FP4 path:
            q_values is packed uint8 and q_scale is the companion
            block-scale tensor.
        kv_cache: Paged KV-cache. FP8 layout is [num_blocks, block_size, D+4]
            or [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. Within
            each block, the D-byte FP8 values for every token are stored first,
            followed by per-token fp32 scale bytes.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.
        clean_logits: Whether to clean the unfilled logits into `-inf`.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    _lazy_init()
    if current_platform.is_device_capability_family(120) and q[1] is None:
        return _fp8_paged_mqa_logits_sm12x(
            q, kv_cache, weights, context_lens, block_tables, max_model_len
        )
    if _fp8_fp4_paged_mqa_logits_impl is None:
        return _missing()
    return _fp8_fp4_paged_mqa_logits_impl(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=clean_logits,
    )


def _tf32_hc_prenorm_gemm_torch(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    """Portable SM12x HyperConnection prenorm GEMM fallback.

    DeepGEMM's split ABI only requires that downstream consumers recover the
    full result by summing over the split dimension. Keep the implementation
    simple by writing the full product to split zero and clearing the rest.
    """
    del num_split
    product = x.float() @ fn.float().T
    norm = x.float().square().sum(dim=-1)

    if out.dim() == 3:
        out.zero_()
        sqrsum.zero_()
        out[0].copy_(product)
        sqrsum[0].copy_(norm)
    else:
        out.copy_(product)
        sqrsum.copy_(norm)
    return out


def _tf32_hc_prenorm_gemm_sm12x(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    if out.dim() == 3 and sqrsum.dim() == 2:
        from vllm.model_executor.layers.deepseek_v4_triton_kernels import (
            tf32_hc_prenorm_gemm_triton,
        )

        tf32_hc_prenorm_gemm_triton(x, fn, out, sqrsum, num_split)
        return out

    return _tf32_hc_prenorm_gemm_torch(x, fn, out, sqrsum, num_split)


def tf32_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    """
    Perform the following computation:
        out = x.float() @ fn.T
        sqrsum = x.float().square().sum(-1)

    See the caller function for shape requirement
    """
    _lazy_init()
    if current_platform.is_device_capability_family(120):
        return _tf32_hc_prenorm_gemm_sm12x(x, fn, out, sqrsum, num_split)
    if _tf32_hc_prenorm_gemm_impl is None:
        return _missing()
    return _tf32_hc_prenorm_gemm_impl(
        x,
        fn,
        out,
        sqrsum,
        num_split,
    )


def _ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _align(x: int, y: int) -> int:
    return cdiv(x, y) * y


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/v2.1.1/csrc/utils/math.hpp#L19
def get_tma_aligned_size(x: int, element_size: int) -> int:
    return _align(x, 16 // element_size)


DEFAULT_BLOCK_SIZE = [128, 128]


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/dd6ed14acbc7445dcef224248a77ab4d22b5f240/deep_gemm/utils/math.py#L38
@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def per_block_cast_to_fp8(
    x: torch.Tensor, block_size: list[int] = DEFAULT_BLOCK_SIZE, use_ue8m0: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = current_platform.fp8_dtype()
    assert x.dim() == 2
    m, n = x.shape
    block_m, block_n = block_size
    x_padded = torch.zeros(
        (_align(m, block_m), _align(n, block_n)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, block_m, x_padded.size(1) // block_n, block_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    _, fp8_max = get_fp8_min_max()
    sf = x_amax / fp8_max
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(fp8_dtype)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    """Return a global difference metric for unit tests.

    DeepGEMM kernels on Blackwell/B200 currently exhibit noticeable per-element
    error, causing `torch.testing.assert_close` to fail.  Instead of checking
    every element, we compute a cosine-style similarity over the whole tensor
    and report `1 - sim`.  Once kernel accuracy improves this helper can be
    removed.
    """

    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def should_use_deepgemm_for_fp8_linear(
    output_dtype: torch.dtype,
    weight_shape: tuple[int, int],
    supports_deep_gemm: bool | None = None,
):
    if supports_deep_gemm is None:
        supports_deep_gemm = is_deep_gemm_supported()

    # Verify DeepGEMM N/K dims requirements
    # NOTE: Also synchronized with test_w8a8_block_fp8_deep_gemm_matmul
    # test inside kernels/quantization/test_block_fp8.py
    N_MULTIPLE = 64
    K_MULTIPLE = 128

    return (
        supports_deep_gemm
        and output_dtype == torch.bfloat16
        and weight_shape[0] % N_MULTIPLE == 0
        and weight_shape[1] % K_MULTIPLE == 0
    )


__all__ = [
    "calc_diff",
    "DeepGemmQuantScaleFMT",
    "fp8_gemm_nt",
    "fp8_einsum",
    "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_fp4_gemm_nt_contiguous",
    "fp8_m_grouped_gemm_nt_masked",
    "fp8_fp4_mqa_logits",
    "fp8_fp4_mqa_topk_indices",
    "fp8_fp4_paged_mqa_logits",
    "fp8_fp4_paged_mqa_topk_indices",
    "get_paged_mqa_logits_metadata",
    "per_block_cast_to_fp8",
    "is_deep_gemm_e8m0_used",
    "is_deep_gemm_supported",
    "get_num_sms",
    "set_num_sms",
    "should_use_deepgemm_for_fp8_linear",
    "get_col_major_tma_aligned_tensor",
    "get_mk_alignment_for_contiguous_layout",
]
