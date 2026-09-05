# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for MultiModalRegistry.supports_multimodal_inputs and
Qwen2.5-VL visual component loading behavior.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.config import SchedulerConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from ..models.utils import build_model_context

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("chunked_prefill", "max_model_len", "expected_seq_len"),
    [(None, 491520, 491520), (True, 491520, 8192), (True, 128, 128), (False, 128, 128)],
)
def test_dummy_inputs_scheduler_budget(
    chunked_prefill, max_model_len, expected_seq_len
):
    model_config = MagicMock()
    model_config.max_model_len = max_model_len
    processor = MagicMock()
    processor.apply.return_value = {"prompt_token_ids": [7]}
    kwargs = {}
    if chunked_prefill is not None:
        kwargs["scheduler_config"] = SchedulerConfig(
            max_model_len=max_model_len,
            is_encoder_decoder=False,
            max_num_batched_tokens=8192,
            max_num_seqs=1,
            enable_chunked_prefill=chunked_prefill,
        )
    result = MULTIMODAL_REGISTRY.get_dummy_mm_inputs(
        model_config, {"image": 1}, processor=processor, **kwargs
    )
    get_inputs = processor.dummy_inputs.get_dummy_processor_inputs
    assert get_inputs.call_args.kwargs["seq_len"] == expected_seq_len
    assert len(result["prompt_token_ids"]) == expected_seq_len


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
