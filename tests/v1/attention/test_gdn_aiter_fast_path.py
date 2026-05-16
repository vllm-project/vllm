# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn_linear_attn import (
    _should_use_aiter_gdn_decode_fast_path,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


def _gdn_decode_metadata(**overrides) -> GDNAttentionMetadata:
    kwargs = dict(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=1,
        spec_sequence_masks=None,
    )
    kwargs.update(overrides)
    return GDNAttentionMetadata(**kwargs)


@pytest.mark.parametrize(
    ("gqa_interleaved_layout", "expected"),
    [
        pytest.param(False, False, id="qwen3_5_non_interleaved_skips"),
        pytest.param(True, True, id="qwen3_next_interleaved_uses"),
    ],
)
def test_aiter_gdn_decode_fast_path_is_layout_gated(
    gqa_interleaved_layout: bool,
    expected: bool,
):
    metadata = _gdn_decode_metadata()

    assert (
        _should_use_aiter_gdn_decode_fast_path(gqa_interleaved_layout, metadata)
        is expected
    )


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(_gdn_decode_metadata(num_prefills=1), id="prefill"),
        pytest.param(
            _gdn_decode_metadata(spec_sequence_masks=torch.tensor([True])),
            id="spec_decode",
        ),
        pytest.param(_gdn_decode_metadata(num_decodes=0), id="no_decodes"),
    ],
)
def test_aiter_gdn_decode_fast_path_still_requires_pure_decode(
    metadata: GDNAttentionMetadata,
):
    assert _should_use_aiter_gdn_decode_fast_path(True, metadata) is False
