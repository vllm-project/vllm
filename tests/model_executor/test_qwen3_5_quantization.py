# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch


def test_qwen3_5_lm_head_receives_quant_config():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.scheduler_config = Mock()
    mock_vllm_config.quant_config = mock_quant_config
    mock_vllm_config.lora_config = None

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5.Qwen3_5Model") as MockModel,
        patch("vllm.model_executor.models.qwen3_5.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.qwen3_5.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        MockModel.return_value.make_empty_intermediate_tensors = Mock()

        Qwen3_5ForCausalLMBase(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def test_qwen3_5_mtp_lm_head_receives_quant_config():
    from vllm.config import CompilationMode
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.compilation_config.mode = CompilationMode.NONE
    mock_vllm_config.quant_config = mock_quant_config

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5_mtp.Qwen3_5MultiTokenPredictor"),
        patch("vllm.model_executor.models.qwen3_5_mtp.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5_mtp.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.qwen3_5_mtp.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        Qwen3_5MTP(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def test_qwen3_5_mtp_forwards_spec_step_idx():
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    mtp = Mock()
    mtp.model.return_value = expected_hidden_states = Mock()
    input_ids = Mock()
    positions = Mock()
    hidden_states = Mock()
    intermediate_tensors = Mock()
    inputs_embeds = Mock()

    output = Qwen3_5MTP.forward(
        mtp,
        input_ids,
        positions,
        hidden_states,
        intermediate_tensors,
        inputs_embeds,
        spec_step_idx=2,
    )

    mtp.model.assert_called_once_with(
        input_ids,
        positions,
        hidden_states,
        intermediate_tensors,
        inputs_embeds,
        2,
    )
    assert output is expected_hidden_states
