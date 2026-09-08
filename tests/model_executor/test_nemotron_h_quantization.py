# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config.mamba import MambaBackendEnum


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
    ("backend", "expected_num_states"),
    [
        (MambaBackendEnum.TRITON, 5),
        (MambaBackendEnum.FLASHINFER, 2),
    ],
)
def test_nemotron_h_replayssm_platform_sizing_is_backend_scoped(
    backend: MambaBackendEnum,
    expected_num_states: int,
):
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

    config = Mock()
    config.cache_config.use_replayssm = True
    config.cache_config.replayssm_buffer_len = 16
    config.cache_config.mamba_cache_dtype = "auto"
    config.cache_config.mamba_ssm_cache_dtype = "float32"
    config.mamba_config.backend = backend
    config.model_config.dtype = torch.bfloat16
    config.model_config.hf_config.mamba_num_heads = 32
    config.model_config.hf_config.mamba_head_dim = 64
    config.model_config.hf_config.n_groups = 8
    config.model_config.hf_config.ssm_state_size = 128
    config.model_config.hf_config.conv_kernel = 4
    config.parallel_config.tensor_parallel_size = 1
    config.num_speculative_tokens = 0

    shapes = NemotronHForCausalLM.get_mamba_state_shape_from_config(config)
    dtypes = NemotronHForCausalLM.get_mamba_state_dtype_from_config(config)

    assert len(shapes) == expected_num_states
    assert len(dtypes) == expected_num_states
