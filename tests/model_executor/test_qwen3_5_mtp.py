# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.5 MTP drafter must not build full-vocabulary tables that the
proposer replaces with the target's after loading (#53887): each one is
vocab_size x hidden_size and can OOM a target that fits without MTP."""

import importlib
from unittest.mock import Mock, patch

import pytest
from torch import nn

# The generic module, and the CUDA one that serves Qwen3.5 MTP on NVIDIA GPUs.
MTP_MODULES = [
    "vllm.model_executor.models.qwen3_5_mtp",
    "vllm.models.qwen3_5.nvidia.mtp",
]

EMBED = "model.language_model.embed_tokens.weight"
LM_HEAD = "lm_head.weight"
MTP_FC = "mtp.fc.weight"


def _import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        pytest.skip(f"{module_name} is unavailable here: {e}")


def _make_vllm_config():
    from vllm.config import CompilationMode

    hf_config = Mock()
    hf_config.tie_word_embeddings = False
    hf_config.vocab_size = 128
    hf_config.hidden_size = 64
    hf_config.num_hidden_layers = 2
    hf_config.mtp_num_hidden_layers = 1
    hf_config.rms_norm_eps = 1e-6

    vllm_config = Mock()
    vllm_config.model_config.hf_text_config = hf_config
    vllm_config.compilation_config.mode = CompilationMode.NONE
    vllm_config.quant_config = None
    return vllm_config


@pytest.mark.parametrize("module_name", MTP_MODULES)
@pytest.mark.parametrize("target_pp", [1, 2])
def test_embedding_built_and_loaded_only_without_target_embedding(
    module_name: str, target_pp: int
):
    """Under a PP>1 target, the draft runs on the last rank, which has no
    target embedding to share, so only then does it keep its own."""
    mtp = _import(module_name)
    own_embed = target_pp > 1
    stubs = {
        name: Mock(return_value=nn.Identity())
        for name in (
            "VocabParallelEmbedding",
            "ColumnParallelLinear",
            "Qwen3_5DecoderLayer",
            "Qwen3_5RMSNorm",
            "get_fused_shared_expert_layers",
        )
        if hasattr(mtp, name)
    }
    with patch.multiple(
        mtp,
        **stubs,
        make_empty_intermediate_tensors_factory=Mock(),
        in_single_pp_stage=Mock(return_value=own_embed),
    ):
        predictor = mtp.Qwen3_5MultiTokenPredictor(vllm_config=_make_vllm_config())
        selector = mtp.Qwen3_5MTP.__new__(mtp.Qwen3_5MTP)
        unused = {
            name: selector.is_unused_checkpoint_weight(name)
            for name in (EMBED, LM_HEAD, MTP_FC)
        }

    assert (predictor.embed_tokens is not None) is own_embed
    assert stubs["VocabParallelEmbedding"].call_count == int(own_embed)
    assert unused == {EMBED: not own_embed, LM_HEAD: True, MTP_FC: False}


@pytest.mark.parametrize("module_name", MTP_MODULES)
def test_lm_head_never_built_or_loaded(module_name: str):
    """The proposer always installs the target's lm_head on the draft."""
    mtp = _import(module_name)
    mock_lm_head = Mock()
    with patch.multiple(
        mtp,
        Qwen3_5MultiTokenPredictor=Mock(return_value=Mock()),
        ParallelLMHead=mock_lm_head,
        LogitsProcessor=Mock(),
        in_single_pp_stage=Mock(return_value=False),
    ):
        model = mtp.Qwen3_5MTP(vllm_config=_make_vllm_config())

        loader = Mock()
        loader.load_weights.side_effect = lambda weights: {n for n, _ in weights}
        with patch.object(mtp, "AutoWeightsLoader", return_value=loader):
            loaded = model.load_weights(
                [(EMBED, Mock()), (LM_HEAD, Mock()), (MTP_FC, Mock())]
            )

    assert model.lm_head is None
    mock_lm_head.assert_not_called()
    assert loaded == {"model.fc.weight"}
