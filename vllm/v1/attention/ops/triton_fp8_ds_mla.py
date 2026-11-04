# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Grouped split-KV attention follows triton_decode_attention.py.
# Software E4M3 conversion follows nickus/vllm-fp8kv (Apache-2.0),
# revision 91aa12ff1b86bd9573090d6ddd5331a90c9c05e1.

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2


@triton.jit
def _dequantize_e4m3(raw, scale):
    b = raw.to(tl.uint16)
    sign = (b & 0x80) << 8
    half = (sign | ((b & 0x7F) << 7)).to(tl.float16, bitcast=True)
    value = half.to(tl.float32) * (scale * 256.0)
    return tl.where((b & 0x7F) == 0x7F, float("nan"), value).to(tl.bfloat16)


@triton.jit
def _decode_fp8_ds_mla_stage1(
    Q,
    Cache,
    Scales,
    Rope,
    Table,
    Lengths,
    Mid,
    Q_BATCH: tl.constexpr,
    Q_HEAD: tl.constexpr,
    TABLE_BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    SPLITS: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CACHE_TOKEN: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    head = (tl.program_id(1) // 2) * 16 + tl.arange(0, 16)
    value_group = tl.program_id(1) % 2
    split = tl.program_id(2)
    hd = tl.arange(0, 128)
    vd = value_group * 256 + tl.arange(0, 256)
    pe = tl.arange(0, 64)
    qbase = batch * Q_BATCH + head[:, None] * Q_HEAD
    qpe = tl.load(Q + qbase + 512 + pe[None, :], head[:, None] < HEADS, 0)
    length = tl.load(Lengths + batch)
    span = tl.cdiv(length, SPLITS)
    begin = split * span
    end = tl.minimum(begin + span, length)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    acc = tl.zeros((16, 256), tl.float32)
    if end > begin:
        for start in range(begin, end, BLOCK_N):
            token = start + tl.arange(0, BLOCK_N)
            page = tl.load(
                Table + batch * TABLE_BATCH + token // PAGE, token < end, 0
            ).to(tl.int64)
            byte_offset = page * CACHE_BLOCK + token % PAGE * CACHE_TOKEN
            logits = tl.zeros((16, BLOCK_N), tl.float32)
            for group in range(4):
                kd = group * 128 + hd
                q = tl.load(Q + qbase + kd[None, :], head[:, None] < HEADS, 0)
                raw = tl.load(
                    Cache + byte_offset[None, :] + kd[:, None], token[None, :] < end, 0
                )
                scales = tl.load(
                    Scales + byte_offset // 4 + 128 + group, token < end, 0
                )
                k = _dequantize_e4m3(raw, scales[None, :])
                logits += tl.dot(q, k)
            kpe = tl.load(
                Rope + byte_offset[None, :] // 2 + 264 + pe[:, None],
                token[None, :] < end,
                0,
            )
            logits = (logits + tl.dot(qpe, kpe)) * SCALE
            logits = tl.where(
                (head[:, None] < HEADS) & (token[None, :] < end), logits, -float("inf")
            )
            new_maximum = tl.maximum(maximum, tl.max(logits, 1))
            alpha = tl.exp(maximum - new_maximum)
            weights = tl.exp(logits - new_maximum[:, None])
            raw_v = tl.load(
                Cache + byte_offset[:, None] + vd[None, :], token[:, None] < end, 0
            )
            groups_v = value_group * 2 + tl.arange(0, 2)
            scales_v = tl.load(
                Scales + byte_offset[:, None] // 4 + 128 + groups_v[None, :],
                token[:, None] < end,
                0,
            )
            raw_v = tl.reshape(raw_v, (BLOCK_N, 2, 128))
            v = _dequantize_e4m3(raw_v, scales_v[:, :, None])
            v = tl.reshape(v, (BLOCK_N, 256))
            acc = acc * alpha[:, None] + tl.dot(weights.to(tl.bfloat16), v)
            denominator = denominator * alpha + tl.sum(weights, 1)
            maximum = new_maximum
        mid_row_width = KV_LORA_RANK + 1
        mid_base = ((batch * HEADS + head[:, None]) * SPLITS + split) * mid_row_width
        tl.store(
            Mid + mid_base + vd[None, :],
            acc / denominator[:, None],
            head[:, None] < HEADS,
        )
        lse_base = (
            (batch * HEADS + head) * SPLITS + split
        ) * mid_row_width + KV_LORA_RANK
        if value_group == 0:
            tl.store(Mid + lse_base, maximum + tl.log(denominator), head < HEADS)


def decode_attention_fwd_fp8_ds_mla(
    q: torch.Tensor,
    cache: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    attn_logits: torch.Tensor,
    num_kv_splits: int,
    sm_scale: float,
) -> None:
    """Decode BF16 queries shaped [batch, heads, 576] over packed MLA pages.

    Cache rows contain 512 E4M3 bytes, four FP32 group scales, and 64 BF16
    RoPE values. Writes [batch, heads, 512] output and [batch, heads] LSE,
    using contiguous FP32 workspace shaped [batch, heads, splits, 513].
    """
    assert q.dtype == torch.bfloat16 and q.shape[-1] == 576
    assert cache.dtype == torch.uint8 and cache.shape[-1] == 656
    assert cache.stride(-1) == 1
    assert (
        cache.stride(0) % 4 == 0
        and cache.stride(1) % 4 == 0
        and cache.storage_offset() % 4 == 0
    ), "fp8_ds_mla cache strides and storage offset must be divisible by 4"
    batch, heads = q.shape[:2]
    kv_lora_rank = o.shape[-1]
    assert kv_lora_rank == 512
    assert (
        attn_logits.dtype == torch.float32
        and attn_logits.shape == (batch, heads, num_kv_splits, kv_lora_rank + 1)
        and attn_logits.is_contiguous()
    ), (
        "fp8_ds_mla workspace must be contiguous FP32 with shape "
        "[batch, heads, splits, rank + 1]"
    )
    _decode_fp8_ds_mla_stage1[(batch, triton.cdiv(heads, 16) * 2, num_kv_splits)](
        q,
        cache,
        cache.view(torch.float32),
        cache.view(torch.bfloat16),
        block_table,
        seq_lens,
        attn_logits,
        q.stride(0),
        q.stride(1),
        block_table.stride(0),
        heads,
        cache.shape[1],
        num_kv_splits,
        cache.stride(0),
        cache.stride(1),
        kv_lora_rank,
        32,
        sm_scale,
        num_warps=4,
        num_stages=2,
    )
    _fwd_kernel_stage2[(batch, heads)](
        attn_logits,
        o,
        lse,
        seq_lens,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        o.stride(0),
        o.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=num_kv_splits,
        BLOCK_DV=kv_lora_rank,
        Lv=kv_lora_rank,
        num_warps=4,
        num_stages=2,
    )
