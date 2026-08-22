# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

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


def test_nemotron_h_quantized_replayssm_config_includes_state_scale():
    from types import SimpleNamespace

    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="int8",
            use_replayssm=True,
            replayssm_buffer_len=16,
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                mamba_num_heads=8,
                mamba_head_dim=4,
                n_groups=2,
                ssm_state_size=16,
                conv_kernel=4,
            ),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        mamba_config=SimpleNamespace(backend=MambaBackendEnum.FLASHINFER),
        num_speculative_tokens=3,
    )

    dtypes = NemotronHForCausalLM.get_mamba_state_dtype_from_config(vllm_config)
    shapes = NemotronHForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    assert dtypes[-1] == torch.float32
    assert shapes[-1] == shapes[1][:-1]
    assert len(shapes) == len(dtypes) == 6
