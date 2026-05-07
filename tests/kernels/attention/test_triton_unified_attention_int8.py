# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

DEVICE_TYPE = current_platform.device_type

NUM_HEADS = [(4, 4), (8, 2)]
HEAD_SIZES = [64, 128]
BLOCK_SIZES = [16]
SEQ_THRESHOLD_3D_VALUES = [0, 8]


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len] * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        mask = torch.triu(
            torch.ones(query_len, kv_len), diagonal=kv_len - query_len + 1
        ).bool()
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@torch.inference_mode()
def test_int8_reshape_and_cache_roundtrip(num_heads: int, head_size: int) -> None:
    """Quantize K/V to int8 then dequantize: max error within half a step."""
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    num_tokens, block_size, num_blocks = 16, 16, 4

    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16) * 2.0
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16) * 2.0

    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=torch.int8
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=torch.int8
    )

    k_scale_val = key.abs().max().float() / 127.0
    v_scale_val = value.abs().max().float() / 127.0
    k_scale = torch.tensor([k_scale_val], dtype=torch.float32)
    v_scale = torch.tensor([v_scale_val], dtype=torch.float32)
    slot_mapping = torch.arange(num_tokens, dtype=torch.long)

    triton_reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        "int8_per_tensor",
        k_scale,
        v_scale,
    )

    key_dequant = key_cache[:1, :num_tokens].float() * k_scale_val
    value_dequant = value_cache[:1, :num_tokens].float() * v_scale_val

    key_err = (key_dequant - key.float()).abs().max()
    value_err = (value_dequant - value.float()).abs().max()

    assert key_err <= k_scale_val * 0.6, (
        f"Key error {key_err:.6f} > {k_scale_val * 0.6:.6f}"
    )
    assert value_err <= v_scale_val * 0.6, (
        f"Value error {value_err:.6f} > {v_scale_val * 0.6:.6f}"
    )


@torch.inference_mode()
def test_int8_reshape_and_cache_scale_one_lossless() -> None:
    """With scale=1.0 and integer inputs in [-127,127], quant must be lossless.

    This is the regression guard for the rounding implementation: any kernel
    edit that drops round-to-nearest-even (e.g. naive ``floor(x + 0.5)``) is
    still lossless on integer inputs, but a non-RNE implementation breaks
    on half-integer inputs handled by ``test_int8_reshape_and_cache_roundtrip``.
    """
    torch.set_default_device(DEVICE_TYPE)

    num_tokens, num_heads, head_size = 8, 4, 64
    block_size, num_blocks = 16, 2

    key = torch.randint(
        -50, 50, (num_tokens, num_heads, head_size), dtype=torch.float16
    )
    value = torch.randint(
        -50, 50, (num_tokens, num_heads, head_size), dtype=torch.float16
    )

    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=torch.int8
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=torch.int8
    )

    k_scale = torch.tensor([1.0], dtype=torch.float32)
    v_scale = torch.tensor([1.0], dtype=torch.float32)
    slot_mapping = torch.arange(num_tokens, dtype=torch.long)

    triton_reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        "int8_per_tensor",
        k_scale,
        v_scale,
    )

    assert torch.equal(key_cache[0, :num_tokens].to(torch.float16), key)
    assert torch.equal(value_cache[0, :num_tokens].to(torch.float16), value)


@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18)], [(1, 523), (1, 2011)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_int8_per_tensor_unified_attention(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    seq_threshold_3D: int,
) -> None:
    """End-to-end INT8 per-tensor attention against pure-pytorch reference.

    Mirrors the FP8 KV-cache pattern in ``test_triton_unified_attention.py``:
    we feed the kernel the int8 cache + per-tensor descales and compare
    against attention computed on the dequantized cache.
    """
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    num_blocks = 2048

    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, dtype=torch.bfloat16
    )
    key_cache_ref = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=torch.bfloat16
    )
    value_cache_ref = torch.randn_like(key_cache_ref)

    k_scale_val = key_cache_ref.abs().max().float().item() / 127.0
    v_scale_val = value_cache_ref.abs().max().float().item() / 127.0

    key_cache_int8 = torch.clamp(
        torch.round(key_cache_ref.float() / k_scale_val), -128, 127
    ).to(torch.int8)
    value_cache_int8 = torch.clamp(
        torch.round(value_cache_ref.float() / v_scale_val), -128, 127
    ).to(torch.int8)

    # Reference computes attention on the dequantized cache (the values the
    # kernel effectively sees after int8 -> bf16 conversion + descale fold).
    key_cache_dequant = key_cache_int8.to(torch.bfloat16) * k_scale_val
    value_cache_dequant = value_cache_int8.to(torch.bfloat16) * v_scale_val

    k_descale = torch.full((num_seqs, num_kv_heads), k_scale_val, dtype=torch.float32)
    v_descale = torch.full((num_seqs, num_kv_heads), v_scale_val, dtype=torch.float32)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (
            seq_threshold_3D,
            num_query_heads,
            num_par_softmax_segments,
            head_size_padded,
        ),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    output = torch.empty_like(query)
    unified_attention(
        q=query,
        k=key_cache_int8,
        v=value_cache_int8,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        kv_quant_mode=KVQuantMode.INT8_PER_TENSOR,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache_dequant,
        value_cache=value_cache_dequant,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
    )

    # Same tolerance as the FP8-quantized branch in the FP8 sibling test.
    torch.testing.assert_close(output, ref_output, atol=1.5e-1, rtol=1.5e-1)
