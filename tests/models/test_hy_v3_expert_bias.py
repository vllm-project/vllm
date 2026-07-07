# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import torch
from transformers import PretrainedConfig

from vllm.config import set_current_vllm_config


def test_hy_v3_expert_bias_dtype():
    # Setup mock parallel state
    mock_tp_group = MagicMock()
    mock_tp_group.world_size = 1
    mock_tp_group.rank = 0

    mock_ep_group = MagicMock()
    mock_ep_group.device_group.size.return_value = 1
    mock_ep_group.rank_in_group = 0

    mock_vllm_config = MagicMock()
    mock_vllm_config.parallel_config.eplb_config.num_redundant_experts = 0
    mock_vllm_config.model_config.model = "tencent/Hy3"
    mock_vllm_config.compilation_config.mode = None
    mock_vllm_config.compilation_config.custom_ops = ["none"]

    # Configure mock pretrained config
    config = PretrainedConfig()
    config.num_experts = 192
    config.num_experts_per_tok = 8
    config.expert_hidden_dim = 2048
    config.hidden_size = 4096
    config.num_shared_experts = 2
    config.route_norm = True
    config.rms_norm_eps = 1e-6
    config.hidden_act = "silu"

    # Under set_default_torch_dtype(torch.bfloat16) to simulate standard serving
    torch.set_default_dtype(torch.bfloat16)
    try:
        with (
            set_current_vllm_config(mock_vllm_config),
            patch(
                "vllm.distributed.parallel_state.get_tp_group",
                return_value=mock_tp_group,
            ),
            patch(
                "vllm.distributed.parallel_state.get_ep_group",
                return_value=mock_ep_group,
            ),
            patch(
                "vllm.model_executor.models.hy_v3.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm.model_executor.models.hy_v3.get_ep_group",
                return_value=mock_ep_group,
            ),
            patch("vllm.model_executor.models.hy_v3.GateLinear"),
            patch("vllm.model_executor.models.hy_v3.FusedMoE"),
        ):
            from vllm.model_executor.models.hy_v3 import HYV3MoEFused

            moe = HYV3MoEFused(config=config)

            # 1. Verify expert_bias is allocated in float32 regardless of default
            assert moe.expert_bias.dtype == torch.float32

            # 2. Verify copying a float32 tensor keeps it in float32
            loaded_weight = torch.randn(config.num_experts, dtype=torch.float32)
            moe.expert_bias.data.copy_(loaded_weight)
            assert moe.expert_bias.dtype == torch.float32
            assert torch.equal(moe.expert_bias.data, loaded_weight)
    finally:
        torch.set_default_dtype(torch.float32)
