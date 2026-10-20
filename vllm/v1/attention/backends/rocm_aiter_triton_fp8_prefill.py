# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 Triton prefill attention for the ROCM_AITER_FA backend on gfx942.

With ``VLLM_ROCM_USE_AITER_TRITON_FP8_PREFILL=1``, eligible prefill and extend
calls run aiter's Triton MHA on FP8 Q/K/V instead of CK. Every other call runs
the original CK call with its arguments unchanged. If the aiter kernel cannot
be loaded or a launch raises, the path disables itself with one warning and
CK is used from then on. An out-of-memory error is the exception: that call
falls back to CK but the path stays enabled.

Numerics: Q, K and V each get ONE scale per call, from the amax over every
token and head in the call. The path is lossy (about 4% relative error per head
against CK on Gaussian inputs) and not batch invariant: a co-batched sequence
with larger values, or one large channel such as a sink token, coarsens the
quantisation of everything else in the call.

aiter behaviour this relies on (0.1.21.post2 and 0.1.24):
- ``descale_q`` is read at ``[b, q_head]`` and ``descale_k/v`` at
  ``[b, kv_head]``, with no bounds check and an implicit unit head stride.
- Causal rows that see no key get LSE 0, not -inf, so causal calls must be
  square.
- There is no head-dim-256 config; the default runs at about 0.29x of CK, so
  ``_TUNED_CONFIG`` is a precondition.
- The FP8 output is float32 and LSE is ``[tokens, heads]``; CK returns the
  input dtype and ``[heads, tokens]``.
"""

import inspect
from collections.abc import Callable
from types import ModuleType
from typing import Any

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger

logger = init_logger(__name__)

# Tile size and AMD knobs must be searched together: a staged search settles on
# 64x64/4 warps/1 stage, which spills 17 registers; this config spills none and
# is 1.06x-1.15x faster on every long-context shape tried.
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

# gfx942 FP8 is e4m3fnuz (max 240), not OCP e4m3fn (max 448).
_FP8_MAX = torch.finfo(torch.float8_e4m3fnuz).max
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

# None until the first call; False when the flag is off or the kernel failed.
_enabled: bool | None = None
_mha: ModuleType | None = None


def _disable(err: BaseException) -> None:
    """Stop using the path for the rest of the process, with one warning."""
    global _enabled
    _enabled = False
    logger.warning(
        "FP8 Triton prefill disabled, using the CK path from now on: %r", err
    )


def _fallback(err: BaseException) -> None:
    """Handle a failed launch: fall back for this call, and decide whether to
    keep trying. An out-of-memory error says only that this call did not fit,
    not that the kernel is broken, so the path survives it. Anything else (a
    Triton compile error, an API change) would repeat on every call."""
    if isinstance(err, torch.OutOfMemoryError):
        logger.warning_once("FP8 Triton prefill fell back to CK for one call: %r", err)
        return
    _disable(err)


def _resolve() -> bool:
    global _enabled, _mha
    _enabled = False
    if not rocm_aiter_ops.is_triton_fp8_prefill_enabled():
        return False
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
        # Resolution failures are permanent: the import or the API is wrong.
        _disable(e)
        return False
    _mha = mha
    _enabled = True
    return True


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
    """Return (eligible, reason). Every refusal falls back to the original call."""
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
        # Identity, not equality: comparing values would need a device sync.
        return False, "causal with separate cu_seqlens_q and cu_seqlens_k"
    return True, ""


def _quant_per_tensor(t: torch.Tensor, n_heads: int, batch: int):
    """Quantise ``t`` with one scale and return it with the contiguous
    ``[batch, n_heads]`` descale the kernel indexes per head. Holds a float32
    copy of ``t`` while it runs; the FP8 result is always contiguous."""
    from aiter import dtypes

    amax = t.abs().max().float()
    scale = (amax / _FP8_MAX).clamp(min=1e-8)
    q = (t.float() / scale).to(dtypes.fp8)
    descale = scale.reshape(1, 1).expand(batch, n_heads).contiguous()
    return q, descale


def maybe_triton_prefill_attn(*, orig: Callable[..., Any], **kwargs: Any) -> Any:
    """Drop-in for ``rocm_aiter_ops.flash_attn_varlen_func`` at the prefill and
    extend call sites. Any call that does not run the Triton kernel returns
    ``orig(**kwargs)``."""
    if not (_enabled or (_enabled is None and _resolve())):
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
    # A kwarg this module has never seen cannot be assumed harmless: the CK
    # kernel may act on it. Refuse rather than silently ignore it.
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
        # Inside the try: each call holds a float32 copy of its input while it
        # runs, so on a long-context prefill this is where an OOM lands.
        q8, dq = _quant_per_tensor(q, n_q_heads, batch)
        k8, dk = _quant_per_tensor(k, n_kv_heads, batch)
        v8, dv = _quant_per_tensor(v, n_kv_heads, batch)
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
