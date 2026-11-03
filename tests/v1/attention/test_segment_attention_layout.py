# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Segment attention layout: segment ids/positions, the FlexAttention encoder
mask, the per-candidate pooler and request validation (CPU)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.exceptions import VLLMValidationError
from vllm.model_executor.layers.pooler.tokwise.heads import _mean_pool_segments
from vllm.pooling_params import PoolingParams, SegmentLayoutParams
from vllm.v1.attention.backends import flex_attention
from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata
from vllm.v1.worker.gpu.model_states.encoder_only import segment_layout_rows

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _cpu_doc_ids(monkeypatch):
    """Build the request lookup on the CPU (the backend stages it on the GPU)."""

    def doc_ids(query_start_loc_cpu, device):
        counts = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        return torch.repeat_interleave(torch.arange(len(counts)), counts)

    monkeypatch.setattr(flex_attention, "_offsets_to_doc_ids_tensor", doc_ids)


def test_segment_layout_rows():
    segment_ids, positions = segment_layout_rows([3, 2, 4])
    assert segment_ids.tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    # Each candidate restarts right after the 3-token prefix.
    assert positions.tolist() == [0, 1, 2, 3, 4, 3, 4, 5, 6]


def _encoder_mask(query_start_loc: list[int], segment_ids: list[int] | None):
    metadata = object.__new__(FlexAttentionMetadata)
    metadata.query_start_loc_cpu = torch.tensor(query_start_loc, dtype=torch.int32)
    metadata.query_start_loc = metadata.query_start_loc_cpu
    metadata.segment_ids = (
        None if segment_ids is None else torch.tensor(segment_ids, dtype=torch.int64)
    )
    mask_mod = metadata.get_bidirectional_mask_mod()
    n = query_start_loc[-1]
    q = torch.arange(n).view(-1, 1).expand(n, n)
    kv = torch.arange(n).view(1, -1).expand(n, n)
    zero = torch.zeros((), dtype=torch.int64)
    return mask_mod(zero, zero, q, kv).to(torch.int64).tolist()


def test_encoder_mask_without_layout_is_per_request_bidirectional():
    assert _encoder_mask([0, 2, 3], None) == [[1, 1, 0], [1, 1, 0], [0, 0, 1]]


def test_encoder_mask_with_segment_layout():
    # Request 0: prefix of 2, candidates of 1 and 2 tokens. Request 1: 2 tokens
    # without a layout (all segment 0, fully bidirectional).
    mask = _encoder_mask([0, 5, 7], [0, 0, 1, 2, 2, 0, 0])
    assert mask == [
        # prefix sees only the prefix
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        # candidate 1 sees the prefix and itself
        [1, 1, 1, 0, 0, 0, 0],
        # candidate 2 sees the prefix and itself, never candidate 1
        [1, 1, 0, 1, 1, 0, 0],
        [1, 1, 0, 1, 1, 0, 0],
        # the other request stays separate and bidirectional
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
    ]


def test_mean_pool_segments_skips_prefix():
    data = torch.arange(12, dtype=torch.float32).view(6, 2)
    means = _mean_pool_segments(data, [2, 1, 3])
    assert means.tolist() == [[4.0, 5.0], [8.0, 9.0]]


@pytest.mark.parametrize("lengths", [[4], [0, 2], [3, -1], [2, 1.5]])
def test_segment_layout_params_reject_bad_lengths(lengths):
    with pytest.raises(VLLMValidationError):
        SegmentLayoutParams(segment_lengths=lengths)


def _model_config(attn_type: str):
    return SimpleNamespace(attn_type=attn_type, pooler_config=None)


def test_segment_layout_needs_token_embed():
    params = PoolingParams(
        task="embed", segment_layout_params=SegmentLayoutParams([2, 1])
    )
    with pytest.raises(VLLMValidationError, match="token_embed"):
        params.verify(_model_config("encoder_only"))


def test_segment_layout_needs_encoder_only_attention():
    params = PoolingParams(
        task="token_embed", segment_layout_params=SegmentLayoutParams([2, 1])
    )
    with pytest.raises(VLLMValidationError, match="encoder-only"):
        params.verify(_model_config("decoder"))
