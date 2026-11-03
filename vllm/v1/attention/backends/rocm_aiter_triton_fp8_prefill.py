# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 Triton prefill attention for ROCM_AITER_FA on gfx942.

Ineligible calls run the original CK call unchanged.
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
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

# aiter has no gfx942 FP8 head-dim-256 config yet; its default is slower than CK.
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


@functools.cache
def _load_kernel() -> ModuleType | None:
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
        logger.warning("FP8 Triton prefill unavailable, using CK: %r", e)
        return None
    return mha


def enabled(
    use_prefill_query_quantization: bool,
    head_size: int,
    kv_cache_dtype: str,
    plain_attention: bool,
) -> bool:
    """Whether a layer's prefill and extend calls use the FP8 Triton path.

    ``plain_attention`` is False for layers with ALiBi, sinks or a sliding window.
    """
    if not use_prefill_query_quantization:
        return False
    from vllm.platforms.rocm import get_cdna_version

    if not (
        get_cdna_version() == 3
        and head_size == _HEAD_DIM
        and kv_cache_dtype in ("fp8", "fp8_e4m3")
        and plain_attention
    ):
        logger.info_once(
            "FP8 prefill attention on ROCM_AITER_FA needs gfx942, head size %d, "
            "an FP8 E4M3 KV cache and no ALiBi, sinks or sliding window; "
            "using CK for this layer",
            _HEAD_DIM,
        )
        return False
    if _load_kernel() is None:
        return False
    logger.info_once("FP8 prefill attention enabled for ROCM_AITER_FA")
    return True


def prefill_attn() -> Callable[..., Any]:
    """Return the varlen attention for an enabled layer's prefill and extend."""
    return functools.partial(
        triton_fp8_prefill_attn, orig=rocm_aiter_ops.flash_attn_varlen_func
    )


def _workspace_shapes(
    num_q_tokens: int, num_kv_tokens: int, num_q_heads: int, num_kv_heads: int
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    fp8 = current_platform.fp8_dtype()
    shapes: list[tuple[tuple[int, ...], torch.dtype]] = [
        ((num_q_tokens, num_q_heads, _HEAD_DIM), fp8)
    ]
    if num_kv_tokens:
        shapes += [((num_kv_tokens, num_kv_heads, _HEAD_DIM), fp8)] * 2
    return shapes


def profile(
    query: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    num_kv_tokens: int,
    num_q_tokens: int,
) -> None:
    """Allocate during the profile run what the widest prefill will use.

    ``query`` gives dtype and device only. Its row count is not the bound: with
    microbatching the profile run splits the batch across ubatches, while a
    later step can run unsplit at the full width on the same workspace slot.
    """
    current_workspace_manager().get_simultaneous(
        *_workspace_shapes(num_q_tokens, num_kv_tokens, num_q_heads, num_kv_heads)
    )
    # aiter allocates an FP32 output inside the call, then it is cast.
    torch.empty(
        (num_q_tokens, num_q_heads, _HEAD_DIM),
        dtype=torch.float32,
        device=query.device,
    ).to(query.dtype)


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
    kv_is_fp8=False,
) -> tuple[bool, str]:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        return False, "q/k/v must be [tokens, heads, head_dim]"
    if q.shape[-1] != _HEAD_DIM:
        return False, f"head_dim {q.shape[-1]} != {_HEAD_DIM}"
    if k.shape[-1] != _HEAD_DIM or v.shape[-1] != _HEAD_DIM:
        return False, "k/v head_dim mismatch"
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False, f"dtype {q.dtype} not bf16/fp16"
    kv_dtype = current_platform.fp8_dtype() if kv_is_fp8 else q.dtype
    if k.dtype != kv_dtype or v.dtype != kv_dtype:
        return False, f"k/v dtype {k.dtype}/{v.dtype} != {kv_dtype}"
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


def _descale(scale: torch.Tensor, n_heads: int, batch: int) -> torch.Tensor:
    # aiter reads descales at [b, head] with an implicit unit head stride.
    return scale.expand(batch, n_heads).contiguous()


def _quant(t: torch.Tensor, buf: torch.Tensor, n_heads: int, batch: int):
    _, scale = ops.scaled_fp8_quant(
        t.reshape(t.shape[0], -1), output=buf.view(t.shape[0], -1)
    )
    return buf, _descale(scale, n_heads, batch)


def triton_fp8_prefill_attn(*, orig: Callable[..., Any], **kwargs: Any) -> Any:
    """Run an eligible call with aiter's Triton MHA on FP8 Q/K/V, else ``orig``.

    ``k``/``v`` are either in ``q``'s dtype, or FP8 with one ``descale_k`` and
    ``descale_v`` scale each, which ``orig`` cannot take.
    """
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
    descale_k=None,
    descale_v=None,
    **extra,
):
    kv_is_fp8 = descale_k is not None
    # CK may act on kwargs this module does not know.
    ok, why = (
        (False, f"unexpected kwargs {sorted(extra)}")
        if extra
        else _eligible(
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
            kv_is_fp8,
        )
    )
    if not ok:
        if kv_is_fp8:
            raise ValueError(f"FP8 K/V call is not eligible for Triton: {why}")
        logger.info_once("FP8 Triton prefill INELIGIBLE (%s); using the CK path", why)
        return orig(**kwargs)

    n_q_heads, n_kv_heads = q.shape[1], k.shape[1]
    batch = cu_seqlens_q.numel() - 1
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5

    bufs = current_workspace_manager().get_simultaneous(
        *_workspace_shapes(
            q.shape[0], 0 if kv_is_fp8 else k.shape[0], n_q_heads, n_kv_heads
        )
    )
    q8, dq = _quant(q, bufs[0], n_q_heads, batch)
    if kv_is_fp8:
        k8, v8 = k, v
        dk = _descale(descale_k, n_kv_heads, batch)
        dv = _descale(descale_v, n_kv_heads, batch)
    else:
        k8, dk = _quant(k, bufs[1], n_kv_heads, batch)
        v8, dv = _quant(v, bufs[2], n_kv_heads, batch)
    mha = _load_kernel()
    assert mha is not None
    o, lse = mha._flash_attn_forward(
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
        # Real lengths come from cu_seqlens; rounding avoids per-length recompiles.
        max_seqlen_q=round_up(int(max_seqlen_q), 16),
        max_seqlen_k=round_up(int(max_seqlen_k), 16),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        descale_q=dq,
        descale_k=dk,
        descale_v=dv,
        sink=None,
        config=dict(_TUNED_CONFIG),
    )[:2]

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
