# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Segment attention layout: segment ids/positions, the FlexAttention encoder
mask, the per-candidate pooler and request validation (CPU)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.exceptions import VLLMValidationError
from vllm.model_executor.layers.pooler.tokwise.heads import _mean_pool_segments
from vllm.pooling_params import PoolingParams, SegmentLayoutParams
from vllm.v1.attention.backends import flex_attention
from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata
from vllm.v1.worker.gpu.model_states import encoder_only
from vllm.v1.worker.gpu.model_states.encoder_only import (
    EncoderOnlyModelState,
    segment_layout_rows,
)

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


def test_layout_positions_written_into_persistent_buffer(monkeypatch):
    """CUDA graphs replay the address of the runner's positions buffer, so the
    restarted positions must be written into that buffer, not returned as a
    new tensor (which only eager mode would read)."""
    # CPU-only test: no pinned staging buffers.
    monkeypatch.setattr(encoder_only, "PIN_MEMORY", False)
    state = object.__new__(EncoderOnlyModelState)
    state.rope_state = None
    state.device = torch.device("cpu")
    state.token_type_ids = {}
    state.segment_layouts = {"layout": segment_layout_rows([2, 1, 2])}
    buffer = torch.arange(16, dtype=torch.int64)
    # Two requests in one padded step of 8 tokens: one plain (3 tokens) and
    # one with a layout (5 tokens).
    input_batch = SimpleNamespace(
        req_ids=["plain", "layout"],
        num_scheduled_tokens=np.array([3, 5]),
        num_computed_tokens_np=np.array([0, 0]),
        num_tokens_after_padding=8,
        positions=buffer[:8],
    )
    buffer[:8] = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])

    model_inputs = state.prepare_inputs(input_batch, req_states=None)

    expected = [0, 1, 2, 0, 1, 2, 2, 3]
    positions = model_inputs["positions"]
    assert positions is not None
    assert positions.data_ptr() == buffer.data_ptr()
    assert buffer[:8].tolist() == expected
    assert buffer[8:].tolist() == list(range(8, 16))


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


def _input_processor(sliding_window: int | None):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.engine.input_processor import InputProcessor

    processor = object.__new__(InputProcessor)
    processor.vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        attention_config=SimpleNamespace(backend=AttentionBackendEnum.FLEX_ATTENTION),
    )
    processor.model_config = SimpleNamespace(get_sliding_window=lambda: sliding_window)
    return processor


def test_segment_layout_must_cover_the_prompt():
    processor = _input_processor(sliding_window=None)
    processor._validate_segment_layout([2, 1, 2], prompt_len=5)
    with pytest.raises(VLLMValidationError, match="sum to 5"):
        processor._validate_segment_layout([2, 1, 2], prompt_len=6)


def test_segment_layout_refuses_sliding_window_models():
    # e.g. ModernBERT: local layers window on packed indices, so a candidate's
    # output would depend on its slot.
    with pytest.raises(VLLMValidationError, match="sliding-window"):
        _input_processor(sliding_window=64)._validate_segment_layout(
            [2, 1, 2], prompt_len=5
        )
