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


def test_dflash_build_context_kv_buffers_raises_on_quantized_dtype():
    """Verify mismatched dtype between norm and weight raises clean error."""
    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.hidden_norm = nn.Module()
    model.hidden_norm.weight = nn.Parameter(torch.ones(16, dtype=torch.bfloat16))

    mock_layer_attn = MagicMock()
    mock_layer_attn.qkv_proj.weight = nn.Parameter(
        torch.ones(16, 16, dtype=torch.float8_e4m3fn)
    )

    with pytest.raises(
        NotImplementedError,
        match="DFlash's fused context-KV projection reads qkv_proj.weight",
    ):
        model._build_context_kv_buffers([mock_layer_attn], has_bias=False)
