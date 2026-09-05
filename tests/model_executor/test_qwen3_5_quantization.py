# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch


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


@pytest.mark.parametrize(
    "incompatible_reason",
    (
        "lora",
        "head_dtype",
        "batch_invariant",
        "nan_diagnostics",
        "adaptive",
        "added_vocab",
        "large_vocab",
    ),
)
def test_qwen3_5_does_not_prepare_hybrid_state_for_incompatible_config(
    monkeypatch, incompatible_reason
):
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    hf_config = SimpleNamespace(
        tie_word_embeddings=False,
        vocab_size=128,
        hidden_size=64,
    )
    model_config = SimpleNamespace(
        hf_text_config=hf_config,
        head_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
    )
    speculative_config = SimpleNamespace(enable_adaptive_verification=False)
    vllm_config = SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        scheduler_config=SimpleNamespace(),
        quant_config=Mock(),
        lora_config=None,
        speculative_config=speculative_config,
    )
    if incompatible_reason == "lora":
        vllm_config.lora_config = object()
    elif incompatible_reason == "head_dtype":
        model_config.head_dtype = torch.float32
    elif incompatible_reason == "batch_invariant":
        monkeypatch.setattr("vllm.envs.VLLM_BATCH_INVARIANT", True)
    elif incompatible_reason == "nan_diagnostics":
        monkeypatch.setattr("vllm.envs.VLLM_COMPUTE_NANS_IN_LOGITS", True)
    elif incompatible_reason == "adaptive":
        speculative_config.enable_adaptive_verification = True
    elif incompatible_reason == "added_vocab":
        # ParallelLMHead reports added embeddings after construction below.
        pass
    elif incompatible_reason == "large_vocab":
        hf_config.vocab_size = 1 << 24

    pp_group = SimpleNamespace(is_last_rank=True)
    with (
        patch("vllm.model_executor.models.qwen3_5.Qwen3_5Model") as MockModel,
        patch("vllm.model_executor.models.qwen3_5.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.qwen3_5.get_pp_group",
            return_value=pp_group,
        ),
    ):
        MockModel.return_value.make_empty_intermediate_tensors = Mock()
        MockLMHead.return_value.num_added_embeddings = (
            1 if incompatible_reason == "added_vocab" else 0
        )

        Qwen3_5ForCausalLMBase(vllm_config=vllm_config)

        assert not MockLMHead.return_value._supports_hybrid_nvfp4_lm_head
