# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for MultiModalRegistry.supports_multimodal_inputs and
Qwen2.5-VL visual component loading behavior.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ..models.utils import build_model_context

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("seq_len", [None, 8192])
@pytest.mark.parametrize("token_count", [3, 9000])
def test_dummy_inputs_sequence_budget_preserves_profiling_default(seq_len, token_count):
    """Warmup may override the budget; profiling keeps full-context padding."""
    model_config = MagicMock()
    model_config.max_model_len = 491520
    processor = MagicMock()
    processor.apply.return_value = {"prompt_token_ids": [7] * token_count}
    kwargs = {} if seq_len is None else {"seq_len": seq_len}
    with patch.object(MULTIMODAL_REGISTRY, "create_processor") as create:
        result = MULTIMODAL_REGISTRY.get_dummy_mm_inputs(
            model_config, {"image": 1}, processor=processor, **kwargs
        )
    expected = model_config.max_model_len if seq_len is None else seq_len
    get_inputs = processor.dummy_inputs.get_dummy_processor_inputs
    assert get_inputs.call_args.kwargs["seq_len"] == expected
    assert get_inputs.call_args.kwargs["mm_options"] is (
        model_config.get_multimodal_config.return_value.limit_per_prompt
    )
    assert result["prompt_token_ids"] == [7] * token_count + [0] * max(
        0, expected - token_count
    )
    create.assert_not_called()
    processor.apply.assert_called_once()
    assert model_config.max_model_len == 491520


@pytest.mark.parametrize(
    "model_id,limit_mm_per_prompt,expected",
    [
        ("Qwen/Qwen2-0.5B-Instruct", {}, False),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {}, True),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {"image": 0, "video": 0}, False),
        ("Qwen/Qwen2.5-VL-3B-Instruct", {"image": 0}, True),
    ],
)
@pytest.mark.core_model
def test_supports_multimodal_inputs(model_id, limit_mm_per_prompt, expected):
    """Test supports_multimodal_inputs returns correct boolean for various
    configs."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )
    assert MULTIMODAL_REGISTRY.supports_multimodal_inputs(ctx.model_config) is expected


def test_create_processor_error_uses_served_model_name():
    model_config = SimpleNamespace(
        is_multimodal_model=False,
        model="/path/to/model/weights",
        served_model_name="friendly-model-name",
    )

    with pytest.raises(
        ValueError,
        match="friendly-model-name is not a multimodal model",
    ):
        MULTIMODAL_REGISTRY.create_processor(model_config)
