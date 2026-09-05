# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest
import torch


def test_nemotron_h_lm_head_receives_quant_config():
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

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


@pytest.mark.parametrize(
    "relu2_enabled,quant_fp8_enabled,sm90_supported,config_supported,tp_size,"
    "input_key_state,expected_fusion",
    [
        (False, False, True, True, 1, "matching", True),
        (True, False, True, True, 1, "matching", False),
        (False, True, True, True, 1, "matching", False),
        (False, False, False, True, 1, "matching", False),
        (False, False, True, False, 1, "matching", False),
        (False, False, True, True, 2, "matching", False),
        (False, False, True, True, 1, "wrong", False),
        (False, False, True, True, 1, "missing", False),
    ],
)
def test_relu2_fp8_fusion_follows_dispatch_constraints(
    relu2_enabled: bool,
    quant_fp8_enabled: bool,
    sm90_supported: bool,
    config_supported: bool,
    tp_size: int,
    input_key_state: str,
    expected_fusion: bool,
):
    from vllm.model_executor.models.nemotron_h import (
        NemotronHMLP,
        kFp8StaticTensorSym,
    )

    act_fn = Mock()
    act_fn.enabled.return_value = relu2_enabled
    quant_fp8 = Mock()
    quant_fp8.enabled.return_value = quant_fp8_enabled
    down_proj = Mock(input_scale=Mock())
    if input_key_state != "missing":
        down_proj.input_quant_key = (
            kFp8StaticTensorSym if input_key_state == "matching" else object()
        )
    else:
        del down_proj.input_quant_key
    is_config_supported = Mock(return_value=config_supported)

    with (
        patch.multiple(
            "vllm.model_executor.models.nemotron_h",
            ColumnParallelLinear=Mock(),
            RowParallelLinear=Mock(return_value=down_proj),
            ReLUSquaredActivation=Mock(return_value=act_fn),
            QuantFP8=quant_fp8,
            is_relu_squared_static_fp8_quant_config_supported=is_config_supported,
            get_tensor_model_parallel_world_size=Mock(return_value=tp_size),
        ),
        patch(
            "vllm.model_executor.models.nemotron_h.current_platform.is_cuda",
            return_value=True,
        ),
        patch(
            "vllm.model_executor.models.nemotron_h.current_platform."
            "has_device_capability",
            return_value=sm90_supported,
        ),
    ):
        mlp = NemotronHMLP(
            config=Mock(),
            hidden_size=64,
            intermediate_size=128,
            enable_relu2_fp8_quant=True,
        )

    act_fn.enabled.assert_called_once_with()
    assert mlp.use_relu2_fp8_quant is expected_fusion


@pytest.mark.parametrize(
    "dtype,expected_fusion",
    [(torch.bfloat16, True), (torch.float16, False)],
)
def test_relu2_fp8_fusion_uses_registry(dtype: torch.dtype, expected_fusion: bool):
    from vllm.model_executor.models.nemotron_h import NemotronHMLP

    projected = torch.empty((1, 1), dtype=dtype)
    activated = Mock()
    fused = Mock()
    act_fn = Mock(return_value=activated)
    down_proj = Mock(side_effect=lambda x: (x, None))

    mlp = NemotronHMLP.__new__(NemotronHMLP)
    torch.nn.Module.__init__(mlp)
    mlp.up_proj = Mock(return_value=(projected, None))
    mlp.down_proj = down_proj
    mlp.act_fn = act_fn
    mlp.use_relu2_fp8_quant = True

    with patch(
        "vllm.model_executor.models.nemotron_h.maybe_fused_act_quant",
        return_value=fused,
    ) as maybe_fused:
        result = mlp(Mock())

    if expected_fusion:
        maybe_fused.assert_called_once_with(act_fn, projected, down_proj)
        act_fn.assert_not_called()
        assert result is fused
    else:
        maybe_fused.assert_not_called()
        act_fn.assert_called_once_with(projected)
        assert result is activated


@pytest.mark.parametrize("lora_enabled", [True, False])
def test_shared_relu2_fp8_fusion_disabled_for_lora(lora_enabled: bool):
    from vllm.model_executor.models.nemotron_h import NemotronHMoE

    config = Mock(
        hidden_size=64,
        mlp_bias=False,
        mlp_hidden_act="relu2",
        moe_intermediate_size=128,
        moe_latent_size=None,
        moe_shared_expert_intermediate_size=128,
        n_group=1,
        n_routed_experts=1,
        n_shared_experts=1,
        norm_topk_prob=True,
        num_experts_per_tok=1,
        routed_scaling_factor=1.0,
        topk_group=1,
    )
    parallel_config = Mock(
        enable_eplb=False,
        use_sequence_parallel_moe=False,
    )
    parallel_config.eplb_config.num_redundant_experts = 0
    ep_group = Mock()
    ep_group.device_group.size.return_value = 1
    vllm_config = Mock(lora_config=Mock() if lora_enabled else None)
    mlp = Mock()

    with patch.multiple(
        "vllm.model_executor.models.nemotron_h",
        get_current_vllm_config_or_none=Mock(return_value=vllm_config),
        get_tensor_model_parallel_world_size=Mock(return_value=1),
        get_ep_group=Mock(return_value=ep_group),
        GateLinear=Mock(),
        NemotronHMLP=mlp,
        FusedMoEFactory=Mock(),
    ):
        NemotronHMoE(config=config, parallel_config=parallel_config)

    assert mlp.call_args.kwargs["enable_relu2_fp8_quant"] is not lora_enabled
