# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest


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
    "relu2_enabled,quant_fp8_enabled,fusion_supported,input_key_state,expected_fusion",
    [
        (False, False, True, "matching", True),
        (True, False, True, "matching", False),
        (False, True, True, "matching", False),
        (False, False, False, "matching", False),
        (False, False, True, "wrong", False),
        (False, False, True, "missing", False),
    ],
)
def test_relu2_fp8_fusion_follows_custom_op_dispatch(
    relu2_enabled: bool,
    quant_fp8_enabled: bool,
    fusion_supported: bool,
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
    fusion = Mock()
    fusion_op = Mock(return_value=fusion)
    fusion_op.is_supported_in_current_config.return_value = fusion_supported

    with (
        patch.multiple(
            "vllm.model_executor.models.nemotron_h",
            ColumnParallelLinear=Mock(),
            RowParallelLinear=Mock(return_value=down_proj),
            ReLUSquaredActivation=Mock(return_value=act_fn),
            QuantFP8=quant_fp8,
            Bf16ReLUSquaredStaticFp8Quant=fusion_op,
            get_tensor_model_parallel_world_size=Mock(return_value=1),
        ),
        patch(
            "vllm.model_executor.models.nemotron_h.current_platform.is_cuda",
            return_value=True,
        ),
    ):
        mlp = NemotronHMLP(
            config=Mock(),
            hidden_size=64,
            intermediate_size=128,
            enable_relu2_fp8_quant=True,
        )

    act_fn.enabled.assert_called_once_with()
    if expected_fusion:
        assert mlp.relu2_fp8_quant is fusion
        fusion_op.assert_called_once_with()
    else:
        assert mlp.relu2_fp8_quant is None
        fusion_op.assert_not_called()


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
