# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from unittest.mock import Mock, patch


def test_qwen3_5_models_use_platform_specific_implementation():
    from vllm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MTP
    from vllm.models.qwen3_5.common.mm_preprocess import Qwen3_5ProcessingInfo
    from vllm.platforms import current_platform

    backend = "amd" if current_platform.is_rocm() else "nvidia"
    assert Qwen3_5ForCausalLM.__module__ == f"vllm.models.qwen3_5.{backend}.model"
    assert Qwen3_5MTP.__module__ == f"vllm.models.qwen3_5.{backend}.mtp"
    assert (
        Qwen3_5ProcessingInfo.__module__ == "vllm.models.qwen3_5.common.mm_preprocess"
    )


def test_qwen3_5_models_do_not_use_torch_compile_wrapper():
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MTP

    model_module = import_module(Qwen3_5ForCausalLM.__module__)
    mtp_module = import_module(Qwen3_5MTP.__module__)
    model_cls = model_module.Qwen3_5Model
    predictor_cls = mtp_module.Qwen3_5MultiTokenPredictor

    assert not issubclass(model_cls, TorchCompileWithNoGuardsWrapper)
    assert not issubclass(predictor_cls, TorchCompileWithNoGuardsWrapper)
    assert not issubclass(Qwen3_5MTP, TorchCompileWithNoGuardsWrapper)


def test_qwen3_5_lm_head_receives_quant_config():
    from vllm.models.qwen3_5 import Qwen3_5ForCausalLM

    model_module = Qwen3_5ForCausalLM.__module__

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
        patch(f"{model_module}.Qwen3_5Model") as MockModel,
        patch(f"{model_module}.ParallelLMHead") as MockLMHead,
        patch(f"{model_module}.LogitsProcessor"),
        patch(
            f"{model_module}.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        MockModel.return_value.make_empty_intermediate_tensors = Mock()

        Qwen3_5ForCausalLM(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def test_qwen3_5_mtp_lm_head_receives_quant_config():
    from vllm.models.qwen3_5 import Qwen3_5MTP

    mtp_module = Qwen3_5MTP.__module__

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.quant_config = mock_quant_config

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch(f"{mtp_module}.Qwen3_5MultiTokenPredictor"),
        patch(f"{mtp_module}.ParallelLMHead") as MockLMHead,
        patch(f"{mtp_module}.LogitsProcessor"),
        patch(
            f"{mtp_module}.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        Qwen3_5MTP(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config
