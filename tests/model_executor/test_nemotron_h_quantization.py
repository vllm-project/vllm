# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch


@pytest.mark.parametrize("bf16_activations", [False, True])
def test_nemotron_h_lm_head_receives_quant_config(bf16_activations):
    """The lm_head gets the quant config, whose FP8 linears run as W8A16 only
    in the bf16_activations mode."""
    from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

    mock_quant_config = ModelOptFp8Config(
        quant_method="FP8",
        is_checkpoint_fp8_serialized=True,
        kv_cache_quant_method=None,
        exclude_modules=[],
    )

    mock_hf_config = Mock()
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64
    mock_hf_config.bf16_activations = bf16_activations

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_config = mock_hf_config
    mock_vllm_config.model_config.dtype = None
    mock_vllm_config.scheduler_config = Mock()
    mock_vllm_config.quant_config = mock_quant_config

    with (
        patch("vllm.model_executor.models.nemotron_h.NemotronHModel") as MockModel,
        patch("vllm.model_executor.models.nemotron_h.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.nemotron_h.LogitsProcessor"),
    ):
        MockModel.return_value.make_empty_intermediate_tensors = Mock()
        MockModel.return_value.has_moe = False

        NemotronHForCausalLM(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config
    assert mock_quant_config.w8a16 is bf16_activations


def test_relu2_fp8_fusion_uses_registry():
    from vllm.model_executor.models.nemotron_h import NemotronHMLP

    projected = torch.empty((1, 1), dtype=torch.bfloat16)
    fused = Mock()
    act_fn = Mock()
    down_proj = Mock(side_effect=lambda x: (x, None))

    mlp = NemotronHMLP.__new__(NemotronHMLP)
    torch.nn.Module.__init__(mlp)
    mlp.up_proj = Mock(return_value=(projected, None))
    mlp.down_proj = down_proj
    mlp.act_fn = act_fn

    with patch(
        "vllm.model_executor.models.nemotron_h.maybe_fused_act_quant",
        return_value=fused,
    ) as maybe_fused:
        result = mlp(Mock())

    maybe_fused.assert_called_once_with(act_fn, projected, down_proj)
    act_fn.assert_not_called()
    assert result is fused


_MIXED_FP8_W4A16 = {
    "quant_method": "modelopt",
    "quant_algo": "MIXED_PRECISION",
    "quantized_layers": {
        "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
        "backbone.layers.1.mixer.experts.0.up_proj": {"quant_algo": "W4A16_NVFP4"},
    },
}


@pytest.mark.parametrize(
    "dtype,kv_cache_dtype,quantization,hf_quant_config,error",
    [
        (torch.bfloat16, "bfloat16", "modelopt_mixed", _MIXED_FP8_W4A16, None),
        (torch.bfloat16, "auto", None, None, None),
        (torch.bfloat16, "fp8", "modelopt_mixed", _MIXED_FP8_W4A16, "BF16 KV"),
        (torch.float16, "bfloat16", "modelopt_mixed", _MIXED_FP8_W4A16, "--dtype"),
        (
            torch.bfloat16,
            "bfloat16",
            "modelopt",
            {"quantization": {"quant_algo": "FP8_PB_WO"}},
            "FP8_PB_WO",
        ),
        (torch.bfloat16, "bfloat16", "fp8", None, "no ModelOpt algo"),
    ],
)
def test_nemotron_h_bf16_activations(
    dtype, kv_cache_dtype, quantization, hf_quant_config, error
):
    """bf16_activations serves FP8 linears as W8A16, so it rejects anything
    that would still quantize activations: an FP8 KV cache, a non-BF16 model
    dtype or other quantized layers."""
    from vllm.model_executor.models import config

    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_ssm_cache_dtype="float32", cache_dtype=kv_cache_dtype
        ),
        model_config=SimpleNamespace(
            dtype=dtype,
            quantization=quantization,
            hf_config=SimpleNamespace(
                bf16_activations=True, quantization_config=hf_quant_config
            ),
        ),
    )
    if error is not None:
        with pytest.raises(ValueError, match=error):
            config.NemotronHForCausalLMConfig.verify_and_update_config(vllm_config)
        return
    config.NemotronHForCausalLMConfig.verify_and_update_config(vllm_config)
