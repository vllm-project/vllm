# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for MiniMax M3 CUTLASS sparse speculative decode."""

import math

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    minimax_m3_sparse_attn_decode,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseTritonImpl,
    select_main_backend_and_impl_cls,
)
from vllm.models.minimax_m3.nvidia.msa_cutlass_sparse_decode import (
    MSACutlassDecodePlanCache,
    MSACutlassSparseDecodeRunner,
    prepare_decode_metadata,
    should_prepare_decode_metadata,
)
from vllm.models.minimax_m3.nvidia.sparse_attention_msa import (
    MiniMaxM3SparseMSABackend,
    MiniMaxM3SparseMSAImpl,
)
from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(100):
    pytest.skip(
        "fmha_sm100 sparse decode requires SM100 (Blackwell).",
        allow_module_level=True,
    )


HEAD_DIM = 128
BLOCK_SIZE = 128
TOPK = 16
QUERY_LEN = 4
SM_SCALE = HEAD_DIM**-0.5


@pytest.mark.parametrize(
    (
        "batch_size",
        "decode_query_len",
        "num_q_heads",
        "num_kv_heads",
        "expected",
    ),
    [
        pytest.param(8, 4, 64, 4, False, id="tp1-below-min-batch"),
        pytest.param(16, 4, 64, 4, True, id="tp1-supported"),
        pytest.param(16, 4, 16, 1, True, id="tp4-min-batch"),
        pytest.param(24, 4, 16, 1, True, id="tp4-intermediate-batch"),
        pytest.param(32, 4, 16, 1, True, id="tp4-supported"),
        pytest.param(16, 2, 64, 4, True, id="tp1-query-len-2"),
        pytest.param(16, 2, 16, 1, True, id="tp4-query-len-2"),
    ],
)
def test_msa_cutlass_decode_static_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    decode_query_len: int,
    expected: bool,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    monkeypatch.setenv("VLLM_MINIMAX_M3_MSA_DECODE_BACKEND", "cutlass")
    assert (
        should_prepare_decode_metadata(
            batch_size,
            decode_query_len,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            kv_cache_dtype="fp8_e4m3",
            page_size=BLOCK_SIZE,
            topk_blocks=TOPK,
        )
        is expected
    )


def test_msa_cutlass_decode_static_dispatch_requires_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MINIMAX_M3_MSA_DECODE_BACKEND", "triton")
    assert not should_prepare_decode_metadata(
        32,
        QUERY_LEN,
        num_q_heads=16,
        num_kv_heads=1,
        kv_cache_dtype="fp8_e4m3",
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
    )


def test_msa_cutlass_decode_static_dispatch_accepts_fp8_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MINIMAX_M3_MSA_DECODE_BACKEND", "cutlass")
    assert should_prepare_decode_metadata(
        32,
        QUERY_LEN,
        num_q_heads=16,
        num_kv_heads=1,
        kv_cache_dtype="fp8",
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
    )


@pytest.mark.parametrize(
    "kv_cache_dtype",
    ["auto", "bfloat16", "float16", "fp8_e5m2"],
)
def test_msa_cutlass_decode_static_dispatch_requires_fp8_e4m3(
    monkeypatch: pytest.MonkeyPatch,
    kv_cache_dtype: str,
) -> None:
    monkeypatch.setenv("VLLM_MINIMAX_M3_MSA_DECODE_BACKEND", "cutlass")
    assert not should_prepare_decode_metadata(
        32,
        QUERY_LEN,
        num_q_heads=16,
        num_kv_heads=1,
        kv_cache_dtype=kv_cache_dtype,
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
    )


def test_msa_cutlass_decode_static_dispatch_requires_sm100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MINIMAX_M3_MSA_DECODE_BACKEND", "cutlass")
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda _: False,
    )
    assert not should_prepare_decode_metadata(
        32,
        QUERY_LEN,
        num_q_heads=16,
        num_kv_heads=1,
        kv_cache_dtype="fp8",
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
    )


def test_msa_backend_owns_msa_metadata_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_cls, impl_cls = select_main_backend_and_impl_cls(
        topk_blocks=TOPK,
        kv_cache_dtype="fp8_e4m3",
        num_kv_heads=1,
    )
    assert backend_cls is MiniMaxM3SparseMSABackend
    assert impl_cls is MiniMaxM3SparseMSAImpl

    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda _: False,
    )
    backend_cls, impl_cls = select_main_backend_and_impl_cls(
        topk_blocks=TOPK,
        kv_cache_dtype="fp8_e4m3",
        num_kv_heads=1,
    )
    assert backend_cls is MiniMaxM3SparseBackend
    assert impl_cls is MiniMaxM3SparseTritonImpl


def _make_topk(seq_lens: list[int], num_kv_heads: int) -> torch.Tensor:
    topk = torch.full(
        (sum(QUERY_LEN for _ in seq_lens), num_kv_heads, TOPK),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    for request, seq_len in enumerate(seq_lens):
        for local_query in range(QUERY_LEN):
            token = request * QUERY_LEN + local_query
            visible_tokens = seq_len - QUERY_LEN + local_query + 1
            visible_pages = math.ceil(visible_tokens / BLOCK_SIZE)
            topk[token, :, :visible_pages] = torch.arange(
                visible_pages, dtype=torch.int32, device="cuda"
            )
    return topk


@pytest.mark.parametrize(
    ("num_q_heads", "num_kv_heads", "num_request_pairs"),
    [
        pytest.param(64, 4, 8, id="tp1"),
        pytest.param(16, 1, 16, id="tp4"),
    ],
)
def test_msa_cutlass_decode_matches_triton_with_interleaved_cache(
    monkeypatch: pytest.MonkeyPatch,
    num_q_heads: int,
    num_kv_heads: int,
    num_request_pairs: int,
) -> None:
    monkeypatch.setenv("VLLM_MINIMAX_M3_MSA_DECODE_BACKEND", "cutlass")
    torch.manual_seed(0)
    seq_lens_list = [257, 513] * num_request_pairs
    seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int32)
    seq_lens = seq_lens_cpu.cuda()
    pages_per_request = [math.ceil(seq_len / BLOCK_SIZE) for seq_len in seq_lens_list]
    num_pages = sum(pages_per_request)
    max_pages = max(pages_per_request)

    block_table = torch.zeros(
        len(seq_lens_list), max_pages, dtype=torch.int32, device="cuda"
    )
    physical_pages = torch.randperm(num_pages, dtype=torch.int32, device="cuda")
    offset = 0
    for request, request_pages in enumerate(pages_per_request):
        block_table[request, :request_pages] = physical_pages[
            offset : offset + request_pages
        ]
        offset += request_pages

    key = (
        torch.randn(
            num_pages,
            num_kv_heads,
            BLOCK_SIZE,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    value = (torch.randn_like(key, dtype=torch.bfloat16) * 0.25).to(torch.float8_e4m3fn)
    kv_cache = torch.cat((key, value), dim=-1)
    assert kv_cache.stride() == (
        num_kv_heads * BLOCK_SIZE * 2 * HEAD_DIM,
        BLOCK_SIZE * 2 * HEAD_DIM,
        2 * HEAD_DIM,
        1,
    )

    num_query_tokens = len(seq_lens_list) * QUERY_LEN
    query = torch.randn(
        num_query_tokens,
        num_q_heads,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    query_fp8 = torch.empty_like(query, dtype=torch.float8_e4m3fn)
    ops.scaled_fp8_quant(
        query.view(num_query_tokens, -1),
        scale=q_scale,
        output=query_fp8.view(num_query_tokens, -1),
    )
    query_dequantized = query_fp8.to(torch.bfloat16) * q_scale

    topk_token_major = _make_topk(seq_lens_list, num_kv_heads)
    expected = torch.empty_like(query)
    minimax_m3_sparse_attn_decode(
        query_dequantized,
        kv_cache,
        topk_token_major.transpose(0, 1),
        block_table,
        seq_lens,
        num_kv_heads,
        SM_SCALE,
        expected,
        QUERY_LEN,
        k_scale=None,
        v_scale=None,
    )

    plan_cache = MSACutlassDecodePlanCache()
    metadata = prepare_decode_metadata(
        block_table,
        seq_lens,
        QUERY_LEN,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        plan_cache=plan_cache,
    )
    assert metadata.page_table.data_ptr() == block_table.data_ptr()
    actual = torch.empty_like(query)
    runner = MSACutlassSparseDecodeRunner()
    used = runner.try_decode(
        query,
        kv_cache,
        topk_token_major,
        seq_lens,
        actual,
        metadata,
        num_kv_heads=num_kv_heads,
        scale=SM_SCALE,
        block_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        decode_query_len=QUERY_LEN,
        q_scale=q_scale,
        q_scale_float=1.0,
        k_scale_float=1.0,
        v_scale_float=1.0,
    )

    assert used
    torch.testing.assert_close(runner._get_query_buffer(query), query_fp8)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    # The same captured plan must remain correct as ragged lengths change.
    updated_seq_lens_list = [129, 385] * num_request_pairs
    seq_lens.copy_(
        torch.tensor(updated_seq_lens_list, dtype=torch.int32, device="cuda")
    )
    topk_token_major.copy_(_make_topk(updated_seq_lens_list, num_kv_heads))
    updated_metadata = prepare_decode_metadata(
        block_table,
        seq_lens,
        QUERY_LEN,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        plan_cache=plan_cache,
    )
    assert updated_metadata.plan is metadata.plan
    assert updated_metadata.page_table.data_ptr() == metadata.page_table.data_ptr()

    minimax_m3_sparse_attn_decode(
        query_dequantized,
        kv_cache,
        topk_token_major.transpose(0, 1),
        block_table,
        seq_lens,
        num_kv_heads,
        SM_SCALE,
        expected,
        QUERY_LEN,
        k_scale=None,
        v_scale=None,
    )
    assert runner.try_decode(
        query,
        kv_cache,
        topk_token_major,
        seq_lens,
        actual,
        updated_metadata,
        num_kv_heads=num_kv_heads,
        scale=SM_SCALE,
        block_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        decode_query_len=QUERY_LEN,
        q_scale=q_scale,
        q_scale_float=1.0,
        k_scale_float=1.0,
        v_scale_float=1.0,
    )
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert runner.try_decode(
            query,
            kv_cache,
            topk_token_major,
            seq_lens,
            actual,
            updated_metadata,
            num_kv_heads=num_kv_heads,
            scale=SM_SCALE,
            block_size=BLOCK_SIZE,
            topk_blocks=TOPK,
            decode_query_len=QUERY_LEN,
            q_scale=q_scale,
            q_scale_float=1.0,
            k_scale_float=1.0,
            v_scale_float=1.0,
        )

    replay_seq_lens_list = [257, 513] * num_request_pairs
    seq_lens.copy_(torch.tensor(replay_seq_lens_list, dtype=torch.int32, device="cuda"))
    topk_token_major.copy_(_make_topk(replay_seq_lens_list, num_kv_heads))
    prepare_decode_metadata(
        block_table,
        seq_lens,
        QUERY_LEN,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        plan_cache=plan_cache,
    )
    minimax_m3_sparse_attn_decode(
        query_dequantized,
        kv_cache,
        topk_token_major.transpose(0, 1),
        block_table,
        seq_lens,
        num_kv_heads,
        SM_SCALE,
        expected,
        QUERY_LEN,
        k_scale=None,
        v_scale=None,
    )
    graph.replay()
    current_platform.synchronize()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
