# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, Mock, patch

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


def _build_mtp_predictor(quant_method_name: str, dynamic: dict):
    """Construct Qwen3_5MultiTokenPredictor with mocked internals and return
    the kwargs `ColumnParallelLinear` (i.e. `self.fc`) was built with.

    Regression coverage for a `dynamic`-exclusion ordering bug:
    `quantization_config.dynamic` entries such as `{"-:*mtp*": {}}` are meant
    to let any non-modelopt_fp4 quantized checkpoint (GPTQ, AWQ, gptq_marlin,
    compressed-tensors, ...) opt individual MTP layers out of quantization
    when the checkpoint stores them as plain BF16.
    `Qwen3_5MultiTokenPredictor.__init__` used to implement this by nulling
    `vllm_config.quant_config` when a matching `dynamic` pattern was found --
    but only *after* `self.fc` had already been constructed with the
    original `quant_config`, so the exclusion never actually reached `fc`
    (only `self.layers`, built afterward, benefited). "awq_marlin" is used
    below as a representative quant method name other than "modelopt_fp4":
    as of vllm-project/vllm#53414, "compressed-tensors" is additionally
    forced unquantized regardless of `dynamic`, which happens to mask this
    ordering bug for compressed-tensors specifically but leaves every other
    non-modelopt_fp4 quant method that relies on `dynamic` exclusion --
    plain GPTQ, AWQ, gptq_marlin, etc -- still broken.
    """
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor

    mock_quant_config = Mock()
    mock_quant_config.get_name.return_value = quant_method_name

    mock_hf_config = Mock()
    mock_hf_config.quantization_config = {"dynamic": dynamic}

    mock_hf_text_config = Mock()
    mock_hf_text_config.num_hidden_layers = 48
    mock_hf_text_config.mtp_num_hidden_layers = 1
    mock_hf_text_config.vocab_size = 128
    mock_hf_text_config.hidden_size = 64
    mock_hf_text_config.rms_norm_eps = 1e-6

    mock_model_config = Mock()
    mock_model_config.hf_text_config = mock_hf_text_config
    mock_model_config.hf_config = mock_hf_config

    mock_vllm_config = Mock()
    mock_vllm_config.model_config = mock_model_config
    mock_vllm_config.quant_config = mock_quant_config
    mock_vllm_config.compilation_config.mode = 0
    mock_vllm_config.cache_config.mamba_cache_mode = "align"

    with (
        patch(
            "vllm.model_executor.models.qwen3_5_mtp.VocabParallelEmbedding",
            MagicMock(),
        ),
        patch("vllm.model_executor.models.qwen3_5_mtp.ColumnParallelLinear") as MockFc,
        patch(
            "vllm.model_executor.models.qwen3_5_mtp.Qwen3_5DecoderLayer",
            Mock(side_effect=lambda *a, **kw: torch.nn.Module()),
        ),
        patch(
            "vllm.model_executor.models.qwen3_5_mtp.Qwen3_5RMSNorm",
            Mock(side_effect=lambda *a, **kw: torch.nn.Module()),
        ),
        patch(
            "vllm.model_executor.models.qwen3_5_mtp."
            "make_empty_intermediate_tensors_factory",
            MagicMock(return_value=MagicMock()),
        ),
    ):
        Qwen3_5MultiTokenPredictor(vllm_config=mock_vllm_config, prefix="mtp")
        assert MockFc.call_count == 1
        return MockFc.call_args.kwargs


def test_qwen3_5_mtp_dynamic_exclusion_reaches_fc():
    """A matching `-:*mtp*` dynamic-exclusion pattern must exclude `fc` from
    quantization for any non-modelopt_fp4 quant method, not just
    compressed-tensors. Regression test for the ordering bug described in
    `_build_mtp_predictor`'s docstring."""
    fc_kwargs = _build_mtp_predictor("awq_marlin", {"-:*mtp*": {}})
    assert fc_kwargs["quant_config"] is None


def test_qwen3_5_mtp_no_dynamic_exclusion_still_quantizes_fc():
    """Sanity: without a matching `dynamic` pattern, fc keeps the model's
    quant_config as before (no regression for the common case)."""
    fc_kwargs = _build_mtp_predictor("awq_marlin", {})
    assert fc_kwargs["quant_config"] is not None


def test_qwen3_5_mtp_modelopt_fp4_unaffected():
    """Sanity: the original NVFP4/modelopt_fp4 workaround (#38650) is
    untouched by this fix -- fc stays unquantized regardless of `dynamic`."""
    fc_kwargs = _build_mtp_predictor("modelopt_fp4", {})
    assert fc_kwargs["quant_config"] is None
