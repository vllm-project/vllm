# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MultiModalRegistry.supports_multimodal_inputs and
Qwen2.5-VL visual component loading behavior.
"""

from types import SimpleNamespace

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

pytestmark = pytest.mark.cpu_test


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
