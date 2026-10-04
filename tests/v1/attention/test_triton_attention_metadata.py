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


def _metadata(query_lens):
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
    )


@pytest.mark.parametrize(
    ("num_speculative_tokens", "parallel_drafting", "max_num_seqs", "capacity"),
    [
        (None, False, 8, 8),
        (4, False, 8, 40),
        (7, False, 8, 64),
        (8, False, 8, 8),
        (3, True, 8, 56),
        (4, True, 8, 8),
        (5, False, 1, 6),
    ],
)
def test_segment_scratch_covers_only_configured_eligible_queries(
    monkeypatch, num_speculative_tokens, parallel_drafting, max_num_seqs, capacity
):
    # 16-segment sizing; SM12.0 rows are covered in test_triton_attn_segments.py.
    monkeypatch.setattr(current_platform, "is_device_capability", lambda _: False)
    builder = _builder(num_speculative_tokens, parallel_drafting, max_num_seqs)
    assert builder.softmax_segm_output.shape == (capacity, 8, 16, 128)
    assert builder.softmax_segm_max.shape == (capacity, 8, 16)
    assert builder.softmax_segm_expsum.shape == (capacity, 8, 16)
