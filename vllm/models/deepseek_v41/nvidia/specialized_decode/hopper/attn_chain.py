# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The early-trigger small kernels of the Hopper attention chain
(``attn_cute``), for steps on the small-batch sparse MLA up to ``MAX_ROWS``
rows.

They replace, with bitwise-equal results:

- ``fused_q_kv_rmsnorm`` and the KV half of
  ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`` by one kernel on the
  ``fused_wqa_wkv`` output (``norm_kv_insert``);
- the Q half of that op by ``q_rope`` (RoPE of the 16 live heads; the small MLA
  reads only those, so the padding heads are left unwritten);
- the MLA ``Combine`` and ``fused_inv_rope_fp8_quant`` by one kernel writing
  the ``wo_a`` input (``mla_small``).

Each triggers its dependents at block start and waits for its predecessor only
before the first activation read, so the next kernel (a GEMM that loads its
weights before its own wait) becomes resident and loads during it.
"""

from functools import cache
from typing import TYPE_CHECKING, Any

import torch

from ...flashmla import DeepseekV4FlashMLAAttention

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

MAX_ROWS = 8
_Q_LORA, _KV, _HEADS, _HEAD_DIM, _ROPE = 1280, 512, 16, 512, 64


def applies(attn: DeepseekV4FlashMLAAttention) -> bool:
    """Whether the layer's shapes and cache format are the kernels': a TP4
    shard on the V4 ``fp8_ds_mla`` SWA record with an FP32 GPT-J cache."""
    cos_sin = attn.rotary_emb.cos_sin_cache
    return (
        attn.q_lora_rank == _Q_LORA
        and attn.head_dim == _HEAD_DIM
        and attn.n_local_heads == _HEADS
        and attn.padded_heads == 64
        and attn.n_local_groups == 2
        and not attn.kv_mxfp8
        and not attn.use_pcp
        and not attn.accepts_unnormed_unroped_query
        and cos_sin.dtype == torch.float32
        and cos_sin.shape[-1] == _ROPE
    )


@cache
def compiled(kind: str, flag: Any):
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass.cute.runtime import make_fake_stream

    from . import attn_cute

    i, p = Int32(1), Int64(0)
    if kind == "norm":
        kernel: Any = attn_cute.QkvNorm(flag)
        args = [p, i, p, i, p, p, p, p, p, i, p, p, p, i]
    elif kind == "qrope":
        kernel = attn_cute.QRope()
        args = [p, i, p, p, p, i]
    else:
        kernel = attn_cute.CombineInvRope(_HEADS, flag)
        args = [p, p, p, p, i, i, p, p, i, i, i]
    return cute.compile(kernel, *args, make_fake_stream())


def launch(kind: str, flag: Any, *args: Any) -> None:
    """``kind``'s kernel on the current stream; tensors pass their address,
    ``("i64", n)`` a 64-bit value, other ints Int32."""
    from cutlass import Int32, Int64

    from ...ops.cute_dsl.primitives import current_cu_stream

    values: list = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            values.append(Int64(arg.data_ptr()))
        elif isinstance(arg, tuple):
            values.append(Int64(arg[1]))
        else:
            values.append(Int32(arg))
    compiled(kind, flag)(*values, current_cu_stream())


def norm_kv_insert(
    attn: DeepseekV4FlashMLAAttention,
    qr_kv: torch.Tensor,
    positions: torch.Tensor,
    swa_metadata: "DeepseekSparseSWAMetadata",
) -> torch.Tensor:
    """The normalized query ``qr`` [rows, 1280] of the ``fused_wqa_wkv``
    output; the step's KV (normalized, rotated, quantized) is inserted into
    the SWA cache."""
    rows = qr_kv.shape[0]
    assert qr_kv.shape[1] == _Q_LORA + _KV and qr_kv.stride(1) == 1
    assert positions.dtype == torch.int64
    slots = swa_metadata.slot_mapping
    assert slots.dtype == torch.int64
    cache = attn.swa_cache_layer.kv_cache
    cache_2d = cache.view(cache.shape[0], -1)
    qr = torch.empty(rows, _Q_LORA, dtype=qr_kv.dtype, device=qr_kv.device)
    launch(
        "norm",
        float(attn.eps),
        qr_kv,
        qr_kv.stride(0),
        qr,
        qr.stride(0),
        ("i64", 0),
        attn.q_norm.weight,
        attn.kv_norm.weight,
        cache_2d,
        ("i64", cache_2d.stride(0)),
        swa_metadata.block_size,
        slots,
        positions,
        attn.rotary_emb.cos_sin_cache,
        rows,
    )
    return qr


def q_rope(
    attn: DeepseekV4FlashMLAAttention, q: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """``q`` [rows, 16, 512] with RoPE on its last 64 dims, in a [rows, 64,
    512] buffer whose padding heads are left unwritten (the small MLA reads
    heads 0..15 only)."""
    rows = q.shape[0]
    assert q.shape[1:] == (_HEADS, _HEAD_DIM) and q.stride(1) == _HEAD_DIM
    out = torch.empty(rows, 64, _HEAD_DIM, dtype=q.dtype, device=q.device)
    launch(
        "qrope",
        None,
        q,
        q.stride(0),
        out,
        positions,
        attn.rotary_emb.cos_sin_cache,
        rows,
    )
    return out


def warm_up(eps: float) -> None:
    """Compile the kernels before CUDA graph capture."""
    compiled("norm", eps)
    compiled("qrope", None)
    for slots in (8, 40):
        compiled("combine", slots)
