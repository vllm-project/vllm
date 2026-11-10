# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The early-trigger q/kv norm of the Hopper attention chain (`attn_cute`),
for uniform steps on the small-batch sparse MLA (`mla_small`).

`norm_kv_insert` replaces `fused_q_kv_rmsnorm` and the KV half of
`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` with bitwise-equal
results by one kernel on the `fused_wqa_wkv` output, which also writes the
step's keys into the decode's SWA tile images
(`mla_small.SmallMlaDecode.chain_norm`). The decode that follows `wq_b`
(`mla_cute.ChainDecode`) takes over the Q half of that op (the query RoPE,
in its prologue) and `fused_inv_rope_fp8_quant`, writing the `wo_a`
input. Each kernel triggers
its dependents at block start and waits for its predecessor only before the
first activation read, so the next kernel (a GEMM that loads its weights before
its own wait) becomes resident and loads during it.
"""

from functools import cache
from typing import TYPE_CHECKING

import torch

from ...flashmla import DeepseekV4FlashMLAAttention
from .mla_small import launch

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

_Q_LORA, _KV, _HEADS, _HEAD_DIM, _ROPE = 1280, 512, 16, 512, 64


def applies(attn: DeepseekV4FlashMLAAttention) -> bool:
    """Whether the layer's shapes and cache format are the kernels': a TP4
    shard on the V4 `fp8_ds_mla` SWA record with an FP32 GPT-J cache."""
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
def _compiled(eps: float):
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass.cute.runtime import make_fake_stream

    from . import attn_cute

    i, p = Int32(1), Int64(0)
    args = [p, i, p, i, p, p, p, p, p, i, p, p, p, p, p, p, i, i, i, i]
    return cute.compile(attn_cute.QkvNorm(eps), *args, make_fake_stream())


def norm_kv_insert(
    attn: DeepseekV4FlashMLAAttention,
    qr_kv: torch.Tensor,
    positions: torch.Tensor,
    swa_metadata: "DeepseekSparseSWAMetadata",
    image: tuple,
) -> torch.Tensor:
    """The normalized query `qr` [rows, 1280] of the `fused_wqa_wkv`
    output; the step's KV (normalized, rotated, quantized) is inserted into
    the SWA cache and into the small MLA's SWA tile images. `image`: the
    images, unions and step shape (`mla_small.SmallMlaDecode.chain_norm`)."""
    rows = qr_kv.shape[0]
    assert qr_kv.shape[1] == _Q_LORA + _KV and qr_kv.stride(1) == 1
    assert positions.dtype == torch.int64
    slots = swa_metadata.slot_mapping
    assert slots.dtype == torch.int64
    cache = attn.swa_cache_layer.kv_cache
    cache_2d = cache.view(cache.shape[0], -1)
    qr = torch.empty(rows, _Q_LORA, dtype=qr_kv.dtype, device=qr_kv.device)
    launch(
        _compiled(float(attn.eps)),
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
        *image,
        rows,
    )
    return qr


def warm_up(eps: float) -> None:
    """Compile the kernels before CUDA graph capture."""
    _compiled(eps)
