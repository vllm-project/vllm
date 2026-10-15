# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LoRA rejection in Zamba2 and Mamba2 models."""

from unittest.mock import MagicMock

import pytest

from vllm.model_executor.models.mamba2 import Mamba2Model
from vllm.model_executor.models.zamba2 import Zamba2Model

pytestmark = pytest.mark.cpu_test


def test_mamba2_raises_value_error_when_lora_enabled():
    vllm_config = MagicMock()
    vllm_config.lora_config = MagicMock()
    with pytest.raises(
        ValueError, match="enable_lora=True is not supported for Mamba2"
    ):
        Mamba2Model(vllm_config=vllm_config)


def test_zamba2_raises_value_error_when_lora_enabled():
    vllm_config = MagicMock()
    vllm_config.lora_config = MagicMock()
    with pytest.raises(
        ValueError, match="enable_lora=True is not supported for Zamba2"
    ):
        Zamba2Model(vllm_config=vllm_config)
