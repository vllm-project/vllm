# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for decoder prefill routed through the zentorch SDPA kernel."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.zentorch_utils import has_zentorch_op
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.zentorch_sdpa import (
    should_use_zentorch_prefill_sdpa,
    zentorch_prefill_sdpa,
)

if not current_platform.is_cpu() or not current_platform.is_zen_cpu():
    pytest.skip("skipping non-Zen CPU tests", allow_module_level=True)

ATOL, RTOL = 1.5e-2, 1e-2

UNIFORM_SEQ_LENS = [128, 128, 128]
RAGGED_SEQ_LENS = [1, 67, 233, 5]


def _metadata(
    query_lens: list[int],
    seq_lens: list[int] | None = None,
    *,
    num_extend_reqs: int = 0,
    num_native_reqs: int = 0,
    num_native_tokens: int = 0,
    block_table: torch.Tensor | None = None,
) -> SimpleNamespace:
    seq_lens = seq_lens or query_lens
    start_loc = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(query_lens, dtype=torch.int32), 0, out=start_loc[1:])
    if block_table is None:
        block_table = torch.zeros(
            (num_native_reqs + len(query_lens), 1), dtype=torch.int32
        )
    return SimpleNamespace(
        prefill_query_start_loc=start_loc,
        seq_lens=torch.tensor(
            [1] * num_native_reqs + list(seq_lens), dtype=torch.int32
        ),
        num_extend_reqs=num_extend_reqs,
        num_native_reqs=num_native_reqs,
        num_native_tokens=num_native_tokens,
        block_table=block_table,
    )


def _ref_varlen_causal_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: list[int],
    scale: float,
    query_lens: list[int] | None = None,
) -> torch.Tensor:
    """Causal attention; query may be a suffix of each sequence (extend)."""
    query_lens = query_lens or seq_lens
    output = torch.empty_like(query)
    q_start = 0
    kv_start = 0
    for query_len, seq_len in zip(query_lens, seq_lens, strict=True):
        q = query[q_start : q_start + query_len].float() * scale
        k = key[kv_start : kv_start + seq_len].float()
        v = value[kv_start : kv_start + seq_len].float()
        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k)
        q_pos = torch.arange(seq_len - query_len, seq_len)[:, None]
        k_pos = torch.arange(seq_len)[None, :]
        attn.masked_fill_(k_pos > q_pos, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("hqk,khd->qhd", attn, v).to(dtype=query.dtype)
        output[q_start : q_start + query_len].copy_(out)
        q_start += query_len
        kv_start += seq_len
    return output


def _qkv(
    seq_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    tokens = sum(seq_lens)
    query = torch.randn(tokens, num_heads[0], head_size, dtype=dtype)
    key = torch.randn(tokens, num_heads[1], head_size, dtype=dtype)
    value = torch.randn(tokens, num_heads[1], head_size, dtype=dtype)
    return query, key, value, torch.empty_like(query)


def _empty_cache(
    key: torch.Tensor, block_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(1, key.size(1), block_size, key.size(2), dtype=key.dtype),
        torch.zeros(1, key.size(1), block_size, key.size(2), dtype=key.dtype),
    )


@pytest.mark.parametrize(
    "attn_type,alibi,window,softcap,fp8,expected",
    [
        (AttentionType.DECODER, None, -1, 0, False, True),
        (AttentionType.ENCODER_ONLY, None, -1, 0, False, False),
        (AttentionType.ENCODER_DECODER, None, -1, 0, False, False),
        (AttentionType.DECODER, torch.ones(4), -1, 0, False, False),
        (AttentionType.DECODER, None, 256, 0, False, False),
        (AttentionType.DECODER, None, -1, 50.0, False, False),
        (AttentionType.DECODER, None, -1, 0, True, False),
    ],
)
def test_should_use_zentorch_prefill_sdpa_gates(
    attn_type: str,
    alibi: torch.Tensor | None,
    window: int,
    softcap: float,
    fp8: bool,
    expected: bool,
) -> None:
    if expected and not has_zentorch_op(["zentorch_sdpa"]):
        expected = False
    if expected and not torch.cpu._is_avx512_bf16_supported():
        expected = False
    assert (
        should_use_zentorch_prefill_sdpa(
            attn_type,
            alibi,
            window,
            softcap,
            None,
            fp8,
            None,
            torch.bfloat16,
        )
        is expected
    )


@pytest.mark.skipif(
    not has_zentorch_op(["zentorch_sdpa"]),
    reason="zentorch_sdpa op not available",
)
@pytest.mark.parametrize("seq_lens", [UNIFORM_SEQ_LENS, RAGGED_SEQ_LENS])
@pytest.mark.parametrize("num_heads", [(8, 2), (4, 4)])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_prefill_sdpa_matches_reference(
    seq_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
) -> None:
    """Uniform batches take the dense path, ragged ones the padded path."""
    set_random_seed(0)
    query, key, value, output = _qkv(seq_lens, num_heads, head_size, dtype)
    scale = head_size**-0.5
    key_cache, value_cache = _empty_cache(key)

    zentorch_prefill_sdpa(
        query,
        key,
        value,
        output,
        key_cache,
        value_cache,
        _metadata(seq_lens),
        scale,
    )

    ref_output = _ref_varlen_causal_attn(query, key, value, seq_lens, scale)
    torch.testing.assert_close(output, ref_output, atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(
    not has_zentorch_op(["zentorch_sdpa"]),
    reason="zentorch_sdpa op not available",
)
@torch.inference_mode()
def test_extend_sdpa_reads_paged_prefix() -> None:
    """Chunked prefill attends cached prefix tokens, not only this chunk."""
    set_random_seed(0)
    seq_len, query_len, block_size = 48, 16, 32
    prefix_len = seq_len - query_len
    num_heads, head_size = 4, 64
    dtype = torch.bfloat16
    scale = head_size**-0.5

    full_q, full_k, full_v, _ = _qkv(
        [seq_len], (num_heads, num_heads), head_size, dtype
    )
    query = full_q[prefix_len:]
    packed_k = full_k[prefix_len:]
    packed_v = full_v[prefix_len:]
    output = torch.empty_like(query)

    num_blocks = (seq_len + block_size - 1) // block_size
    key_cache = torch.zeros(num_blocks, num_heads, block_size, head_size, dtype=dtype)
    value_cache = torch.zeros_like(key_cache)
    for token in range(seq_len):
        block, offset = divmod(token, block_size)
        key_cache[block, :, offset] = full_k[token]
        value_cache[block, :, offset] = full_v[token]
    block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

    zentorch_prefill_sdpa(
        query,
        packed_k,
        packed_v,
        output,
        key_cache,
        value_cache,
        _metadata(
            [query_len],
            [seq_len],
            num_extend_reqs=1,
            block_table=block_table,
        ),
        scale,
    )

    ref = _ref_varlen_causal_attn(
        query, full_k, full_v, [seq_len], scale, query_lens=[query_len]
    )
    torch.testing.assert_close(output, ref, atol=ATOL, rtol=RTOL)
