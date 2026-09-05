# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Triton causal-prefill launch-order classification."""

from typing import cast
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


def _create_builder() -> TritonAttentionMetadataBuilder:
    builder = object.__new__(TritonAttentionMetadataBuilder)
    builder.device = DEVICE
    builder.seq_threshold_3D = 0
    builder.num_par_softmax_segments = 1
    builder.softmax_segm_output = torch.empty(0)
    builder.softmax_segm_max = torch.empty(0)
    builder.softmax_segm_expsum = torch.empty(0)
    builder.rswa_window = None
    builder.persistent_rswa_prefix_lens = None
    builder.reorder_causal_prefill = True
    return builder


def _build_metadata(
    seq_lens: list[int],
    query_lens: list[int],
    is_prefilling: list[bool] | None,
) -> TritonAttentionMetadata:
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=BLOCK_SIZE,
        device=DEVICE,
    )
    if is_prefilling is not None:
        common.is_prefilling = torch.tensor(is_prefilling, dtype=torch.bool)
    return _create_builder().build(0, common)


@pytest.mark.parametrize(
    ("seq_lens", "query_lens", "is_prefilling", "expected_reorder"),
    [
        pytest.param([8, 12], [1, 1], [False, False], False, id="regular_decode"),
        pytest.param([8, 12], [4, 4], [False, False], False, id="spec_decode_only"),
        pytest.param([32], [32], [True], True, id="prefill"),
        pytest.param(
            [8, 32], [4, 32], [False, True], True, id="mixed_spec_and_prefill"
        ),
        pytest.param([33], [1], [True], False, id="final_single_token_prefill"),
        pytest.param([8], [4], None, False, id="phase_metadata_unavailable"),
    ],
)
def test_triton_metadata_computes_final_prefill_reorder_decision(
    seq_lens: list[int],
    query_lens: list[int],
    is_prefilling: list[bool] | None,
    expected_reorder: bool,
) -> None:
    metadata = _build_metadata(seq_lens, query_lens, is_prefilling)

    assert metadata.reorder_causal_prefill is expected_reorder


@pytest.mark.parametrize(
    ("reorder_enabled", "query_len"),
    [
        pytest.param(False, 4, id="feature_disabled"),
        pytest.param(True, 1, id="single_token_query"),
    ],
)
def test_triton_metadata_skips_unneeded_phase_reduction(
    reorder_enabled: bool,
    query_len: int,
) -> None:
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[8], query_lens=[query_len]),
        block_size=BLOCK_SIZE,
        device=DEVICE,
    )
    is_prefilling = MagicMock()
    common.is_prefilling = cast(torch.Tensor, is_prefilling)
    builder = _create_builder()
    builder.reorder_causal_prefill = reorder_enabled

    metadata = builder.build(0, common)

    assert metadata.reorder_causal_prefill is False
    is_prefilling.__getitem__.assert_not_called()
    is_prefilling.any.assert_not_called()


def test_triton_cudagraph_capture_disables_prefill_reorder() -> None:
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[32], query_lens=[32]),
        block_size=BLOCK_SIZE,
        device=DEVICE,
    )
    common.is_prefilling = torch.tensor([True])

    metadata = _create_builder().build_for_cudagraph_capture(common)

    assert metadata.reorder_causal_prefill is False
