# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest
from torch import nn


def _make_vllm_config():
    from vllm.config import CompilationMode

    hf_config = Mock()
    hf_config.tie_word_embeddings = False
    hf_config.vocab_size = 128
    hf_config.hidden_size = 64
    hf_config.num_hidden_layers = 2
    hf_config.mtp_num_hidden_layers = 1
    hf_config.rms_norm_eps = 1e-6
    hf_config.num_experts = 0

    vllm_config = Mock()
    vllm_config.model_config.hf_text_config = hf_config
    vllm_config.cache_config.mamba_cache_mode = "align"
    vllm_config.compilation_config.mode = CompilationMode.NONE
    vllm_config.quant_config = None
    return vllm_config


@pytest.mark.parametrize(("pp_size", "should_allocate"), [(1, False), (2, True)])
def test_qwen3_5_mtp_embedding_allocation_depends_on_pp(
    pp_size: int, should_allocate: bool
):
    from vllm.model_executor.models import qwen3_5_mtp

    mock_embedding = Mock(return_value=nn.Identity())
    with patch.multiple(
        qwen3_5_mtp,
        VocabParallelEmbedding=mock_embedding,
        ColumnParallelLinear=Mock(return_value=nn.Identity()),
        Qwen3_5DecoderLayer=Mock(return_value=nn.Identity()),
        Qwen3_5RMSNorm=Mock(return_value=nn.Identity()),
        get_pp_group=Mock(return_value=Mock(world_size=pp_size)),
        is_model_fused_shared_expert_compatible=Mock(return_value=False),
        make_empty_intermediate_tensors_factory=Mock(),
    ):
        predictor = qwen3_5_mtp.Qwen3_5MultiTokenPredictor(
            vllm_config=_make_vllm_config()
        )

    assert (predictor.embed_tokens is not None) is should_allocate
    assert mock_embedding.call_count == int(should_allocate)


def test_qwen3_5_mtp_skips_shared_vocab_modules_and_weights_for_pp1():
    from vllm.model_executor.models import qwen3_5_mtp

    mock_lm_head = Mock()
    with patch.multiple(
        qwen3_5_mtp,
        Qwen3_5MultiTokenPredictor=Mock(return_value=nn.Module()),
        ParallelLMHead=mock_lm_head,
        LogitsProcessor=Mock(),
        get_pp_group=Mock(return_value=Mock(world_size=1, is_last_rank=True)),
    ):
        model = qwen3_5_mtp.Qwen3_5MTP(vllm_config=_make_vllm_config())

    assert model.lm_head is None
    mock_lm_head.assert_not_called()

    loader = Mock()
    loader.load_weights.side_effect = lambda weights: {name for name, _ in weights}
    weights = [
        ("language_model.model.embed_tokens.weight", Mock()),
        ("lm_head.weight", Mock()),
        ("mtp.fc.weight", Mock()),
    ]
    with patch(
        "vllm.model_executor.models.qwen3_5_mtp.AutoWeightsLoader",
        return_value=loader,
    ):
        assert model.load_weights(weights) == {"model.fc.weight"}
