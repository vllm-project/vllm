# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config import LoadConfig, ModelConfig, SpeculativeConfig, VllmConfig
from vllm.model_executor.models.utils import get_draft_quant_config
from vllm.platforms import current_platform

DEVICES = (
    [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]
    if current_platform.is_cuda_alike()
    else ["cpu"]
)


def test_get_draft_quant_config_with_draft_model():
    mock_draft_model_config = Mock(spec=ModelConfig)
    mock_load_config = Mock(spec=LoadConfig)
    mock_speculative_config = Mock(spec=SpeculativeConfig)
    mock_speculative_config.draft_model_config = mock_draft_model_config

    mock_vllm_config = Mock(spec=VllmConfig)
    mock_vllm_config.speculative_config = mock_speculative_config
    mock_vllm_config.load_config = mock_load_config

    mock_quant_config = Mock()
    with patch.object(
        VllmConfig, "get_quantization_config", return_value=mock_quant_config
    ):
        result = get_draft_quant_config(mock_vllm_config)

        # Verify the function calls get_quantization_config with draft model config
        VllmConfig.get_quantization_config.assert_called_once_with(
            mock_draft_model_config, mock_load_config
        )
        assert result == mock_quant_config


def test_get_draft_quant_config_without_draft_model():
    mock_speculative_config = Mock(spec=SpeculativeConfig)
    mock_speculative_config.draft_model_config = None

    mock_vllm_config = Mock(spec=VllmConfig)
    mock_vllm_config.speculative_config = mock_speculative_config
    mock_vllm_config.load_config = Mock(spec=LoadConfig)

    result = get_draft_quant_config(mock_vllm_config)

    assert result is None


@torch.inference_mode()
@pytest.mark.parametrize("device", DEVICES)
def test_fc_layer_quant_config_usage(default_vllm_config, dist_init, device) -> None:
    import torch

    from vllm.model_executor.layers.linear import ReplicatedLinear

    if current_platform.is_cuda_alike():
        torch.cuda.set_device(device)

    torch.set_default_device(device)

    input_size = 256
    output_size = 128

    fc_no_quant = ReplicatedLinear(
        input_size=input_size,
        output_size=output_size,
        bias=False,
        params_dtype=torch.float16,
        quant_config=None,
        prefix="fc",
    )

    assert fc_no_quant.quant_config is None
    assert fc_no_quant.input_size == input_size
    assert fc_no_quant.output_size == output_size

    mock_quant_config = Mock()
    fc_with_quant = ReplicatedLinear(
        input_size=input_size,
        output_size=output_size,
        bias=False,
        params_dtype=torch.float16,
        quant_config=mock_quant_config,
        prefix="fc",
    )

    assert fc_with_quant.quant_config == mock_quant_config

    # Check forward pass
    x = torch.randn(2, input_size, dtype=torch.float16)
    output, _ = fc_no_quant(x)
    assert output.shape == (2, output_size)


def test_kv_cache_scale_name_handling():
    # Mock a quant config that supports cache scales
    mock_quant_config = Mock()
    mock_quant_config.get_cache_scale = Mock(return_value="layers.0.self_attn.kv_scale")

    # Condition check in load_weights
    name = "layers.0.self_attn.k_proj.weight"
    scale_name = mock_quant_config.get_cache_scale(name)

    # Check if get_cache_scale is called and returns expected value
    mock_quant_config.get_cache_scale.assert_called_once_with(name)
    assert scale_name == "layers.0.self_attn.kv_scale"


def test_kv_cache_scale_name_no_scale():
    # Mock a quant config that returns None for get_cache_scale
    mock_quant_config = Mock()
    mock_quant_config.get_cache_scale = Mock(return_value=None)

    name = "layers.0.mlp.gate_proj.weight"
    scale_name = mock_quant_config.get_cache_scale(name)

    # Should return None for weights that don't have cache scales
    assert scale_name is None


def test_maybe_remap_kv_scale_name():
    from vllm.model_executor.model_loader.weight_utils import maybe_remap_kv_scale_name

    params_dict = {
        "layers.0.self_attn.kv_scale": Mock(),
        "layers.1.self_attn.kv_scale": Mock(),
    }

    name = "layers.0.self_attn.some_scale"
    remapped = maybe_remap_kv_scale_name(name, params_dict)

    assert remapped in params_dict or remapped == name or remapped is None


def test_load_weights_kv_scale_handling():
    kv_scale_param = Mock()
    kv_scale_param.weight_loader = Mock()

    params_dict = {
        "layers.0.self_attn.kv_scale": kv_scale_param,
    }

    mock_quant_config = Mock()
    mock_quant_config.get_cache_scale = Mock(return_value="layers.0.self_attn.kv_scale")

    # Load_weights logic for KV cache scales
    name = "layers.0.self_attn.k_proj.weight"
    loaded_weight_tensor = torch.tensor([1.0, 2.0])

    if mock_quant_config is not None:
        scale_name = mock_quant_config.get_cache_scale(name)
        if scale_name:
            param = params_dict[scale_name]
            assert param is kv_scale_param
            weight_to_load = (
                loaded_weight_tensor
                if loaded_weight_tensor.dim() == 0
                else loaded_weight_tensor[0]
            )

            assert scale_name == "layers.0.self_attn.kv_scale"
            assert weight_to_load == loaded_weight_tensor[0]
