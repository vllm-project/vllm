# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LoRA rejection in Jamba model."""

from unittest.mock import MagicMock

import pytest

from vllm.model_executor.models.jamba import JambaModel

pytestmark = pytest.mark.cpu_test


def test_jamba_raises_value_error_when_lora_enabled():
    vllm_config = MagicMock()
    vllm_config.lora_config = MagicMock()
    with pytest.raises(
        ValueError, match="enable_lora=True is not supported for Jamba"
    ):
        JambaModel(vllm_config=vllm_config)
