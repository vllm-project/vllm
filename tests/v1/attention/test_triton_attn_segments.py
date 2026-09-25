# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
from vllm.v1.attention.backends.triton_attn_diffkv import (
    TritonAttentionDiffKVMetadataBuilder,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec


# 64 segments only while num_seqs * num_kv_heads * 16 < 188 SMs on SM12.0.
@pytest.mark.parametrize(
    "builder_cls",
    [TritonAttentionMetadataBuilder, TritonAttentionDiffKVMetadataBuilder],
)
@pytest.mark.parametrize(
    "capability,num_kv_heads,num_seqs,segments",
    [
        ((12, 0), 1, 11, 64),
        ((12, 0), 1, 12, 16),
        ((12, 0), 8, 1, 64),
        ((12, 0), 8, 2, 16),
        ((9, 0), 1, 1, 16),
    ],
)
def test_split_k_segments_follow_sm_occupancy(
    monkeypatch, builder_cls, capability, num_kv_heads, num_seqs, segments
):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda cap: cap == capability
    )
    monkeypatch.setattr(current_platform, "num_compute_units", lambda: 188)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _: 8,
            get_num_kv_heads=lambda _: num_kv_heads,
            get_head_size=lambda: 128,
            rswa_window=None,
        ),
        parallel_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=16),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE, static_forward_context={}
        ),
    )
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=num_kv_heads, head_size=128, dtype=torch.bfloat16
    )
    builder = builder_cls(spec, ["layer.0"], config, "cpu")
    batch = BatchSpec(seq_lens=[128] * num_seqs, query_lens=[1] * num_seqs)
    metadata = builder.build(0, create_common_attn_metadata(batch, 16, "cpu"))
    assert metadata.num_par_softmax_segments == segments
    assert metadata.softmax_segm_output.shape[2] == segments
    # DiffKV re-allocates the output; it must stay sized like the base buffers.
    assert builder.softmax_segm_output.shape[0] == builder.softmax_segm_max.shape[0]
    assert (
        metadata.softmax_segm_output.data_ptr()
        == builder.softmax_segm_output.data_ptr()
    )
