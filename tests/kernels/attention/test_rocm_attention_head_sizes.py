# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm paged-attention and Triton decode head-size coverage."""

import math

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm.platforms import current_platform
from vllm.utils.torch_utils import create_kv_caches_with_random, set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

DTYPES = [torch.bfloat16, torch.float16]
NUM_HEADS_PAIRS = [(16, 16), (16, 4)]
NUM_BLOCKS = 2048
BLOCK_SIZE = 16
ALIBI_NUM_HEADS_PAIRS = [(8, 8), (16, 4)]
ROCM_PAGED_ATTN_SUPPORTED_HEAD_SIZES = [64, 128]
ROCM_PAGED_ATTN_UNSUPPORTED_HEAD_SIZES = [32, 80, 96, 160, 192, 224, 256]
TRITON_DECODE_HEAD_SIZES = [32, 64, 80, 96, 128, 192, 256]
DECODE_SEQ_LENS = [128, 256, 384, 512]


def ref_block_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    alibi_slopes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference paged attention using naive einsum implementation."""
    num_seqs = len(query_lens)
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    device = query.device
    slopes = None if alibi_slopes is None else alibi_slopes.to(device)[:, None, None]

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q = q * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        mask = torch.triu(
            torch.ones(query_len, kv_len, device=device),
            diagonal=kv_len - query_len + 1,
        ).bool()
        attn.masked_fill_(mask, float("-inf"))

        if slopes is not None:
            q_positions = torch.arange(
                kv_len - query_len, kv_len, dtype=torch.float32, device=device
            )
            k_positions = torch.arange(kv_len, dtype=torch.float32, device=device)
            position_bias = q_positions[:, None] - k_positions[None, :]
            attn += -slopes.float() * position_bias

        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    return torch.einsum("hqk,khd->qhd", attn_weights, value)


def ref_rocm_paged_attn(
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    alibi_slopes: torch.Tensor | None,
) -> torch.Tensor:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()
    outputs: list[torch.Tensor] = []

    for i, seq_len in enumerate(seq_lens_lst):
        q = query[i].unsqueeze(0)
        block_table = block_tables_lst[i]

        keys_lst: list[torch.Tensor] = []
        values_lst: list[torch.Tensor] = []
        for j in range(int(seq_len)):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            keys_lst.append(k.reshape(num_kv_heads, head_size))

            v = value_cache[block_number, :, :, block_offset]
            values_lst.append(v)

        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        if num_queries_per_kv > 1:
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            position_ids = torch.arange(seq_len, device=query.device).int()
            alibi_bias = (position_ids - seq_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        outputs.append(out.view(num_query_heads, head_size))

    return torch.stack(outputs)


def _get_alibi_slopes(num_heads: int, device: torch.device) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
        device=device,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32, device=device)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
            device=device,
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(
            start=1,
            end=1 + 2 * num_remaining_heads,
            step=2,
            dtype=torch.int32,
            device=device,
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes.float()


def _run_paged_attention_rocm(
    head_size: int,
    dtype: torch.dtype,
    num_heads: tuple[int, int],
    *,
    seq_lens: list[int] = DECODE_SEQ_LENS,
    alibi_slopes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ROCm paged attention and return kernel and reference outputs."""
    from vllm import _custom_ops as ops

    num_query_heads, num_kv_heads = num_heads
    num_seqs = len(seq_lens)
    max_seq_len = max(seq_lens)
    scale = head_size**-0.5

    query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    query.uniform_(-scale, scale)

    key_caches, value_caches = create_kv_caches_with_random(
        NUM_BLOCKS,
        BLOCK_SIZE,
        1,
        num_kv_heads,
        head_size,
        "auto",
        dtype,
        0,
        "cuda",
    )
    key_cache = key_caches[0]
    value_cache = value_caches[0]

    max_num_blocks_per_seq = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)

    output = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    num_partitions = (max_seq_len + 255) // 256
    tmp_output = torch.empty(
        num_seqs,
        num_query_heads,
        num_partitions,
        head_size,
        dtype=torch.float32,
    )
    exp_sums = torch.empty(
        num_seqs, num_query_heads, num_partitions, dtype=torch.float32
    )
    max_logits = torch.empty_like(exp_sums)
    scale_tensor = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    ops.paged_attention_rocm(
        output,
        exp_sums,
        max_logits,
        tmp_output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens_tensor,
        None,
        BLOCK_SIZE,
        max_seq_len,
        alibi_slopes,
        "auto",
        scale_tensor,
        scale_tensor,
    )

    ref_output = ref_rocm_paged_attn(
        query,
        num_query_heads // num_kv_heads,
        key_cache,
        value_cache,
        block_tables,
        seq_lens_tensor,
        scale,
        alibi_slopes,
    )
    return output, ref_output


@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_heads", ALIBI_NUM_HEADS_PAIRS)
def test_rocm_paged_attn_alibi(head_size, dtype, num_heads):
    """ROCm paged attention with ALiBi matches the naive reference."""
    torch.set_default_device("cuda")
    set_random_seed(0)

    num_query_heads, num_kv_heads = num_heads
    seq_lens = [64, 128, 192, 256]
    alibi_slopes = _get_alibi_slopes(num_query_heads, device=torch.device("cuda"))
    output, ref_output = _run_paged_attention_rocm(
        head_size=head_size,
        dtype=dtype,
        num_heads=(num_query_heads, num_kv_heads),
        seq_lens=seq_lens,
        alibi_slopes=alibi_slopes,
    )
    torch.testing.assert_close(
        output,
        ref_output,
        atol=get_default_atol(output),
        rtol=get_default_rtol(output),
    )


@pytest.mark.parametrize("head_size", ROCM_PAGED_ATTN_SUPPORTED_HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_heads", NUM_HEADS_PAIRS)
def test_rocm_paged_attn_head_sizes(head_size, dtype, num_heads):
    """ROCm paged attention matches the naive reference on supported head sizes."""
    torch.set_default_device("cuda")
    set_random_seed(0)

    output, ref_output = _run_paged_attention_rocm(
        head_size=head_size,
        dtype=dtype,
        num_heads=num_heads,
    )
    torch.testing.assert_close(
        output,
        ref_output,
        atol=get_default_atol(output),
        rtol=get_default_rtol(output),
    )


@pytest.mark.parametrize("head_size", ROCM_PAGED_ATTN_UNSUPPORTED_HEAD_SIZES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_heads", [(16, 16)])
def test_rocm_paged_attn_unsupported_head_sizes(head_size, dtype, num_heads):
    """Verify ROCm paged attention rejects unsupported head sizes."""
    torch.set_default_device("cuda")
    set_random_seed(0)

    with pytest.raises(RuntimeError, match="Unsupported head size"):
        _run_paged_attention_rocm(
            head_size=head_size,
            dtype=dtype,
            num_heads=num_heads,
        )


@pytest.mark.parametrize("head_size", TRITON_DECODE_HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_heads", NUM_HEADS_PAIRS)
def test_triton_attn_decode_head_sizes(head_size, dtype, num_heads):
    """Test Triton decode attention accuracy across head sizes."""
    from vllm.v1.attention.ops.triton_decode_attention import (
        decode_attention_fwd,
    )

    torch.set_default_device("cuda")
    set_random_seed(0)

    num_query_heads, num_kv_heads = num_heads
    batch_size = len(DECODE_SEQ_LENS)

    query = torch.randn(batch_size, num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    scale = head_size**-0.5
    max_kv_len = max(DECODE_SEQ_LENS)
    max_num_blocks_per_seq = (max_kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.randint(
        0, NUM_BLOCKS, (batch_size, max_num_blocks_per_seq), dtype=torch.int32
    )
    seq_lens_tensor = torch.tensor(DECODE_SEQ_LENS, dtype=torch.int32)

    output = torch.zeros(batch_size, num_query_heads, head_size, dtype=dtype)
    lse = torch.zeros(batch_size, num_query_heads, dtype=dtype)

    num_kv_splits = 4
    attn_logits = torch.empty(
        batch_size,
        num_query_heads,
        num_kv_splits,
        head_size + 1,
        dtype=torch.float32,
    )

    # Triton decode attention uses combined KV cache [blocks, page, heads, d]
    # but key and value as separate caches for the reference
    # The kernel expects req_to_token = block_table
    decode_attention_fwd(
        query,
        key_cache,
        value_cache,
        output,
        lse,
        block_tables,
        seq_lens_tensor,
        attn_logits,
        num_kv_splits,
        scale,
        BLOCK_SIZE,
    )

    ref_output = ref_block_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[1] * batch_size,
        kv_lens=DECODE_SEQ_LENS,
        block_tables=block_tables,
        scale=scale,
    )

    atol = 3e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=1e-3)
