# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 Triton prefill attention for ROCM_AITER_FA on gfx942.

Ineligible calls and failed launches run the original CK call unchanged.
One FP8 scale per tensor per call, so results are not batch invariant.
"""

import functools
import inspect
from collections.abc import Callable
from types import ModuleType
from typing import Any

import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import is_quantized_kv_cache

logger = init_logger(__name__)

# aiter has no gfx942 FP8 head-dim-256 config yet.
_TUNED_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 64,
    "PRELOAD_V": True,
    "waves_per_eu": 2,
    "num_warps": 8,
    "num_ctas": 1,
    "num_stages": 2,
    "matrix_instr_nonkdim": 16,
    "kpack": 1,
}

_HEAD_DIM = 256
_KERNEL_KWARGS = frozenset(
    {
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "max_seqlen_k",
        "descale_q",
        "descale_k",
        "descale_v",
        "return_lse",
        "sink",
        "config",
    }
)

# None: not loaded yet. False: disabled for the rest of the process.
_kernel_ok: bool | None = None
_mha: ModuleType | None = None


def _disable(err: BaseException) -> None:
    global _kernel_ok
    _kernel_ok = False
    logger.warning(
        "FP8 Triton prefill disabled, using the CK path from now on: %r", err
    )


def _fallback(err: BaseException) -> None:
    if isinstance(err, torch.OutOfMemoryError):
        logger.warning_once("FP8 Triton prefill fell back to CK for one call: %r", err)
        return
    _disable(err)


def _load_kernel() -> bool:
    global _kernel_ok, _mha
    if _kernel_ok is None:
        try:
            import aiter.ops.triton.attention.mha as mha

            params = inspect.signature(mha._flash_attn_forward).parameters
            missing = _KERNEL_KWARGS - params.keys()
            if missing:
                raise TypeError(f"_flash_attn_forward has no {sorted(missing)}")
            impl = getattr(mha, "_MHA_IMPL", "default")
            if impl != "default":
                raise RuntimeError(f"aiter MHA impl {impl!r} has no FP8 path")
        except Exception as e:
            _disable(e)
            return False
        _mha, _kernel_ok = mha, True
    return _kernel_ok


def prefill_attn(
    use_prefill_query_quantization: bool, head_size: int, kv_cache_dtype: str
) -> Callable[..., Any]:
    """Return the varlen attention for a layer's prefill and extend calls."""
    orig = rocm_aiter_ops.flash_attn_varlen_func
    if not use_prefill_query_quantization:
        return orig
    from vllm.platforms.rocm import get_cdna_version

    if not (
        get_cdna_version() == 3
        and head_size == _HEAD_DIM
        and is_quantized_kv_cache(kv_cache_dtype)
    ):
        logger.info_once(
            "FP8 prefill attention on ROCM_AITER_FA needs gfx942, head size %d "
            "and an FP8 KV cache; using CK",
            _HEAD_DIM,
        )
        return orig
    if not _load_kernel():
        return orig
    logger.info_once("FP8 prefill attention enabled for ROCM_AITER_FA")
    return functools.partial(triton_fp8_prefill_attn, orig=orig)


def _eligible(
    q,
    k,
    v,
    alibi_slopes,
    sink_ptr,
    window_size,
    dropout_p,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
) -> tuple[bool, str]:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        return False, "q/k/v must be [tokens, heads, head_dim]"
    if q.shape[-1] != _HEAD_DIM:
        return False, f"head_dim {q.shape[-1]} != {_HEAD_DIM}"
    if k.shape[-1] != _HEAD_DIM or v.shape[-1] != _HEAD_DIM:
        return False, "k/v head_dim mismatch"
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False, f"dtype {q.dtype} not bf16/fp16"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return False, f"k/v dtype {k.dtype}/{v.dtype} != q dtype {q.dtype}"
    if k.shape[1] != v.shape[1]:
        return False, f"num_k_heads {k.shape[1]} != num_v_heads {v.shape[1]}"
    if q.shape[1] % k.shape[1] != 0:
        return False, (
            f"num_q_heads {q.shape[1]} is not a multiple of num_kv_heads {k.shape[1]}"
        )
    if q.shape[0] == 0 or k.shape[0] == 0 or k.shape[0] != v.shape[0]:
        return False, f"token counts q={q.shape[0]} k={k.shape[0]} v={v.shape[0]}"
    for cu in (cu_seqlens_q, cu_seqlens_k):
        if cu is not None and cu.dtype != torch.int32:
            return False, f"cu_seqlens dtype {cu.dtype} != int32"
    if alibi_slopes is not None:
        return False, "alibi not supported"
    if sink_ptr is not None:
        return False, "attention sink not supported"
    if window_size is not None and tuple(window_size) != (-1, -1):
        return False, f"sliding window {tuple(window_size)} not supported"
    if dropout_p:
        return False, "dropout not supported"
    if causal and cu_seqlens_k is not cu_seqlens_q:
        # aiter writes LSE 0 for causal rows with no key; a value check would sync.
        return False, "causal with separate cu_seqlens_q and cu_seqlens_k"
    return True, ""


def _quant(t: torch.Tensor, n_heads: int, batch: int):
    t8, scale = ops.scaled_fp8_quant(t.reshape(t.shape[0], -1))
    # aiter reads descales at [b, head] with an implicit unit head stride.
    return t8.view(t.shape), scale.expand(batch, n_heads).contiguous()


def triton_fp8_prefill_attn(*, orig: Callable[..., Any], **kwargs: Any) -> Any:
    """Run an eligible call with aiter's Triton MHA on FP8 Q/K/V, else ``orig``."""
    if not _kernel_ok:
        return orig(**kwargs)
    return _triton_or_orig(orig, kwargs, **kwargs)


def _triton_or_orig(
    orig: Callable[..., Any],
    kwargs: dict[str, Any],
    *,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    min_seqlen_q=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    alibi_slopes=None,
    return_lse=False,
    out=None,
    sink_ptr=None,
    **extra,
):
    # CK may act on kwargs this module does not know.
    if extra:
        logger.info_once(
            "FP8 Triton prefill INELIGIBLE (unexpected kwargs %s); using the CK path",
            str(sorted(extra)),
        )
        return orig(**kwargs)
    ok, why = _eligible(
        q,
        k,
        v,
        alibi_slopes,
        sink_ptr,
        window_size,
        dropout_p,
        causal,
        cu_seqlens_q,
        cu_seqlens_k,
    )
    if not ok:
        logger.info_once("FP8 Triton prefill INELIGIBLE (%s); using the CK path", why)
        return orig(**kwargs)

    n_q_heads, n_kv_heads = q.shape[1], k.shape[1]
    batch = cu_seqlens_q.numel() - 1
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5

    assert _mha is not None
    try:
        q8, dq = _quant(q, n_q_heads, batch)
        k8, dk = _quant(k, n_kv_heads, batch)
        v8, dv = _quant(v, n_kv_heads, batch)
        o, lse = _mha._flash_attn_forward(
            q8,
            k8,
            v8,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=bool(causal),
            window_size_left=-1,
            window_size_right=-1,
            bias=None,
            alibi_slopes=None,
            return_lse=bool(return_lse),
            return_softmax=False,
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_seqlen_k),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            descale_q=dq,
            descale_k=dk,
            descale_v=dv,
            sink=None,
            config=dict(_TUNED_CONFIG),
        )[:2]
    except Exception as e:
        _fallback(e)
        return orig(**kwargs)

    logger.info_once(
        "FP8 Triton prefill ACTIVE hd=%d Hq=%d Hkv=%d causal=%s return_lse=%s",
        q.shape[-1],
        n_q_heads,
        n_kv_heads,
        bool(causal),
        bool(return_lse),
    )
    o = out.copy_(o) if out is not None else o.to(q.dtype)
    if not return_lse:
        return o
    return o, lse.transpose(0, 1)
