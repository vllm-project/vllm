# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for MiniCPMV's multimodal preprocessing."""

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["openbmb/MiniCPM-V-4"])
def test_get_hf_processor_for_different_kwargs(model_id: str):
    """Calls with different kwargs must not reuse stale processor instances."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    info = processor.info
    processor_1 = info.get_hf_processor(max_slice_nums=1)
    processor_2 = info.get_hf_processor(max_slice_nums=2)
    assert processor_1.image_processor.max_slice_nums == 1
    assert processor_2.image_processor.max_slice_nums == 2
    assert processor_1 is not processor_2


@pytest.mark.parametrize("model_id", ["openbmb/MiniCPM-V-4"])
def test_get_hf_processor_for_same_kwargs(model_id: str):
    """Same kwargs should return the cached processor instance."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    info = processor.info
    processor_a = info.get_hf_processor(max_slice_nums=1)
    processor_b = info.get_hf_processor(max_slice_nums=1)
    assert processor_a is processor_b
    assert processor_a.image_processor.max_slice_nums == 1
