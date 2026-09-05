# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch


def _make_qwen_gdn_vllm_config():
    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64
    mock_hf_config.num_hidden_layers = 0
    mock_hf_config.layer_types = []
    mock_hf_config.rms_norm_eps = 1e-6

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.parallel_config.eplb_config.num_redundant_experts = 0
    mock_vllm_config.quant_config = mock_quant_config

    return mock_vllm_config, mock_quant_config


def test_qwen3_next_embedding_receives_quant_config():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.models.qwen3_next import Qwen3NextModel

    mock_vllm_config, mock_quant_config = _make_qwen_gdn_vllm_config()
    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        set_current_vllm_config(VllmConfig()),
        patch(
            "vllm.model_executor.models.qwen3_next.VocabParallelEmbedding"
        ) as MockEmbedding,
        patch(
            "vllm.model_executor.models.qwen3_next.make_layers",
            return_value=(0, 0, []),
        ),
        patch(
            "vllm.model_executor.models.qwen3_next."
            "make_empty_intermediate_tensors_factory",
            return_value=Mock(),
        ),
        patch(
            "vllm.model_executor.models.qwen3_next.Qwen3NextRMSNorm",
            return_value=Mock(),
        ),
        patch(
            "vllm.model_executor.models.qwen3_next.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        Qwen3NextModel(vllm_config=mock_vllm_config, prefix="model")

        MockEmbedding.assert_called_once()
        call_kwargs = MockEmbedding.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config
        assert call_kwargs["prefix"] == "model.embed_tokens"


def test_qwen3_5_embedding_receives_quant_config():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.models.qwen3_5 import Qwen3_5Model

    mock_vllm_config, mock_quant_config = _make_qwen_gdn_vllm_config()
    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        set_current_vllm_config(VllmConfig()),
        patch(
            "vllm.model_executor.models.qwen3_5.VocabParallelEmbedding"
        ) as MockEmbedding,
        patch(
            "vllm.model_executor.models.qwen3_5.make_layers",
            return_value=(0, 0, []),
        ),
        patch(
            "vllm.model_executor.models.qwen3_5."
            "make_empty_intermediate_tensors_factory",
            return_value=Mock(),
        ),
        patch(
            "vllm.model_executor.models.qwen3_5.Qwen3_5RMSNorm",
            return_value=Mock(),
        ),
        patch(
            "vllm.model_executor.models.qwen3_5.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        Qwen3_5Model(vllm_config=mock_vllm_config, prefix="model")

        MockEmbedding.assert_called_once()
        call_kwargs = MockEmbedding.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config
        assert call_kwargs["prefix"] == "model.embed_tokens"


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
