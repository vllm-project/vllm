# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import FullAttentionSpec


def _builder(
    num_speculative_tokens=4,
    parallel_drafting=False,
    enable_adaptive_verification=False,
    max_num_seqs=8,
    capture_sizes=None,
    device="cpu",
):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _: 8,
            get_num_kv_heads=lambda _: 1,
            get_head_size=lambda: 128,
            rswa_window=None,
        ),
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=(
            SimpleNamespace(
                num_speculative_tokens=num_speculative_tokens,
                parallel_drafting=parallel_drafting,
                enable_adaptive_verification=enable_adaptive_verification,
            )
            if num_speculative_tokens is not None
            else None
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=(
                CUDAGraphMode.FULL_DECODE_ONLY if capture_sizes else CUDAGraphMode.NONE
            ),
            cudagraph_capture_sizes=capture_sizes or [],
            static_forward_context={},
        ),
    )
    return TritonAttentionMetadataBuilder(
        FullAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
        ),
        ["layer.0"],
        config,
        torch.device(device),
    )


def _metadata(query_lens, is_prefilling):
    starts = torch.tensor([0, *query_lens], dtype=torch.int32, device="cpu")
    starts = starts.cumsum(0, dtype=torch.int32)
    num_reqs = len(query_lens)
    return CommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=torch.tensor(
            [128 if length else 0 for length in query_lens],
            dtype=torch.int32,
            device="cpu",
        ),
        num_reqs=num_reqs,
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        max_seq_len=128,
        block_table_tensor=torch.zeros((num_reqs, 1), dtype=torch.int32, device="cpu"),
        slot_mapping=torch.zeros(sum(query_lens), dtype=torch.int64, device="cpu"),
        is_prefilling=(
            torch.tensor(is_prefilling, dtype=torch.bool, device="cpu")
            if is_prefilling is not None
            else None
        ),
    )


@pytest.mark.parametrize(
    ("query_lens", "is_prefilling", "expected"),
    [
        ([5, 5, 0], [False, False, True], True),
        ([1, 1, 0], [False, False, True], False),
        ([5, 5, 0], [False, False], True),
        ([5, 5, 0], [False], False),
        ([0, 5, 0], [True, False], True),
        ([5, 4, 0], [False, False, False], False),
        ([5, 5, 0], [False, True, False], False),
        ([5, 5, 0], None, False),
        ([0, 0], [False, False], False),
    ],
)
def test_uniform_decode_requires_same_width_nonprefill_rows(
    query_lens, is_prefilling, expected
):
    metadata = _builder().build(0, _metadata(query_lens, is_prefilling))
    assert metadata.is_uniform_decode is expected


def test_uniform_decode_uses_only_current_requests_and_consistent_maximum():
    common = _metadata([5, 5, 0, 42], [False, False, True, True])
    common.num_reqs = 3
    common.num_actual_tokens = 10
    common.max_query_len = 5
    builder = _builder()
    assert builder.build(0, common).is_uniform_decode
    common.max_query_len = 4
    assert not builder.build(0, common).is_uniform_decode


def test_uniform_decode_rejects_adaptive_query_lengths_below_upper_bound():
    common = _metadata([4, 4, 0], [False, False])
    # Adaptive verification evenly splits the CPU budget, then redistributes
    # it on device; max_query_len retains the original scheduled upper bound.
    common.query_start_loc = torch.tensor([0, 3, 8, 8], dtype=torch.int32, device="cpu")
    common.max_query_len = 5
    assert not _builder().build(0, common).is_uniform_decode


def test_adaptive_verification_does_not_admit_cpu_only_uniform_proof():
    common = _metadata([5, 5, 0], [False, False, False])
    builder = _builder(enable_adaptive_verification=True)
    assert not builder.build(0, common).is_uniform_decode


@pytest.mark.parametrize(
    ("num_speculative_tokens", "parallel_drafting", "capacity"),
    [(None, False, 8), (4, False, 40), (2, True, 40), (5, False, 8), (3, True, 8)],
)
def test_segment_scratch_covers_only_configured_eligible_queries(
    num_speculative_tokens, parallel_drafting, capacity
):
    builder = _builder(num_speculative_tokens, parallel_drafting)
    assert builder.softmax_segm_output.shape == (capacity, 8, 16, 128)
    assert builder.softmax_segm_max.shape == (capacity, 8, 16)
    assert builder.softmax_segm_expsum.shape == (capacity, 8, 16)


@pytest.mark.parametrize("query_len", [1, 5])
def test_graph_dummy_lengths_cover_queries_and_reuse_segment_buffers(query_len):
    builder = _builder()
    common = _metadata([query_len, query_len, 0], [False, False, True])
    first = builder.build(0, common)
    captured = builder.build_for_cudagraph_capture(common)
    assert captured.seq_lens.tolist() == [query_len] * 3
    assert captured.softmax_segm_output is first.softmax_segm_output
    assert captured.softmax_segm_max is first.softmax_segm_max
    assert captured.softmax_segm_expsum is first.softmax_segm_expsum


@pytest.fixture
def sm120_platform(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda cap: cap == (12, 0)
    )


@pytest.mark.parametrize(
    "draft_tokens,max_seqs,capacity", [(4, 1, 5), (5, 1, 6), (5, 2, 2)]
)
def test_six_query_capacity_is_single_request_only(
    sm120_platform, draft_tokens, max_seqs, capacity
):
    builder = _builder(draft_tokens, max_num_seqs=max_seqs, capture_sizes=[6])
    assert builder.seq_threshold_3D == 6
    assert builder.softmax_segm_output.shape == (capacity, 8, 16, 128)
    assert builder.softmax_segm_max.shape == (capacity, 8, 16)
    assert builder.softmax_segm_expsum.shape == (capacity, 8, 16)


@pytest.mark.parametrize(
    "query_lens,actual_tokens,prefilling,expected",
    [
        ([6], 6, [False], True),
        ([6], 8, [False], False),
        ([6, 0], 6, [False, False], False),
        ([6], 6, [True], False),
        ([6], 6, None, False),
        ([7], 7, [False], False),
    ],
)
def test_six_query_requires_exact_unpadded_nonprefill_request(
    sm120_platform, query_lens, actual_tokens, prefilling, expected
):
    common = _metadata(query_lens, prefilling)
    common.num_actual_tokens = actual_tokens
    builder = _builder(5, max_num_seqs=1, capture_sizes=[6])
    assert builder.build(0, common).is_uniform_decode is expected


def test_six_query_rejects_shifted_offsets_and_adaptive_metadata(sm120_platform):
    builder = _builder(5, max_num_seqs=1, capture_sizes=[6])
    common = _metadata([6], [False])
    common.query_start_loc_cpu += 1
    assert not builder.build(0, common).is_uniform_decode
    adaptive = _builder(
        5, enable_adaptive_verification=True, max_num_seqs=1, capture_sizes=[6]
    )
    assert not adaptive.build(0, _metadata([6], [False])).is_uniform_decode


def test_six_query_capture_preserves_six_rows_and_buffer_identity(sm120_platform):
    builder = _builder(5, max_num_seqs=1, capture_sizes=[6])
    common = _metadata([6], [False])
    first = builder.build(0, common)
    captured = builder.build_for_cudagraph_capture(common)
    assert captured.is_uniform_decode
    assert captured.seq_lens.tolist() == [6]
    assert captured.softmax_segm_output is first.softmax_segm_output
    assert captured.softmax_segm_max is first.softmax_segm_max
    assert captured.softmax_segm_expsum is first.softmax_segm_expsum
