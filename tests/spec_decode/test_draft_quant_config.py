# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.model_executor.models.utils import get_draft_quant_config


def test_get_draft_quant_config_populates_packed_modules():
    """Verify configure_quant_config is called on the draft model class."""
    mock_vllm_config = MagicMock()
    mock_draft_model_cfg = MagicMock()
    mock_vllm_config.speculative_config.draft_model_config = mock_draft_model_cfg
    mock_vllm_config.load_config = MagicMock()

    mock_quant_config = MagicMock()
    mock_draft_cls = MagicMock()

    with (
        patch(
            "vllm.config.VllmConfig.get_quantization_config",
            return_value=mock_quant_config,
        ),
        patch(
            "vllm.model_executor.models.utils.get_model_architecture",
            return_value=(mock_draft_cls, None),
        ),
        patch(
            "vllm.model_executor.models.utils.configure_quant_config"
        ) as mock_configure,
    ):
        res = get_draft_quant_config(mock_vllm_config)

        assert res == mock_quant_config
        mock_configure.assert_called_once_with(mock_quant_config, mock_draft_cls)


def test_dflash_dequant_kv_slice_raises_value_error_without_scale():
    """Verify that _dequant_kv_slice raises a ValueError if dtypes mismatch
    and no scale exists."""

    class MockQKV(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(10, 10, dtype=torch.float16)

    class MockAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = MockQKV()
            self.q_size = 5

    attn = MockAttn()

    # Passing bfloat16 as the target act_dtype while the weight is float16
    with pytest.raises(ValueError, match="exposes no weight_scale"):
        DFlashQwen3Model._dequant_kv_slice(attn, act_dtype=torch.bfloat16)


def test_dflash_dequant_kv_slice_returns_untouched_on_match():
    """Verify that _dequant_kv_slice returns the tensor directly if dtypes match."""

    class MockQKV(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.ones(10, 10, dtype=torch.bfloat16)

    class MockAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = MockQKV()
            self.q_size = 5

    attn = MockAttn()

    # Target act_dtype matches the weight dtype exactly
    result = DFlashQwen3Model._dequant_kv_slice(attn, act_dtype=torch.bfloat16)

    # Should slice correctly [5:] and return
    assert result.shape == (5, 10)
    assert result.dtype == torch.bfloat16
