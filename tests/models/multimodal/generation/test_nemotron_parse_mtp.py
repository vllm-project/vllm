# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contracts for the Parse 3.5 one-step auxiliary proposer."""

import weakref
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from transformers import PreTrainedConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.nemotron_parse import MBartDecoderNoPos
from vllm.model_executor.models.nemotron_parse_mtp import (
    ParseAuxMTPModel,
    _ParseAuxMTPInner,
)


def test_parse_mtp_config_routes_only_causal_prefix_architecture() -> None:
    config = PreTrainedConfig(
        architectures=["RadioMBartPrefixForCausalLM"],
        decoder=PreTrainedConfig(causal_vision_prefix=True, vocab_size=8, d_model=4),
    )

    draft = SpeculativeConfig.hf_config_override(config)

    assert draft.architectures == ["ParseAuxMTPModel"]
    assert draft.model_type == "mtp"
    assert draft.n_predict == draft.num_nextn_predict_layers == 1


def test_parse_mtp_loads_auxiliary_weights_from_model_file(tmp_path: Path) -> None:
    draft = ParseAuxMTPModel.__new__(ParseAuxMTPModel)
    nn.Module.__init__(draft)
    draft.model = _ParseAuxMTPInner(vocab_size=4, hidden_size=3)

    weights = {
        "decoder.extra_heads.0.weight": torch.arange(9).reshape(3, 3).float(),
        "decoder.extra_heads.0.bias": torch.arange(3).float(),
        "decoder.extra_proj.0.weight": torch.arange(9).reshape(3, 3).float(),
        "decoder.extra_proj.0.bias": torch.arange(3).float(),
        "decoder.embed_tokens.weight": torch.arange(12).reshape(4, 3).float(),
    }
    model_file = tmp_path / "model.safetensors"
    save_file(weights, model_file)

    with safe_open(model_file, framework="pt", device="cpu") as source:
        tensor_names = source.keys()
        loaded = draft.load_weights(
            (name, source.get_tensor(name)) for name in tensor_names
        )

    assert loaded == {
        "model.embed_tokens.weight",
        "model.extra_head.weight",
        "model.extra_head.bias",
        "model.extra_proj.weight",
        "model.extra_proj.bias",
    }
    assert torch.equal(
        draft.model.embed_tokens.weight, weights["decoder.embed_tokens.weight"]
    )
    assert torch.equal(
        draft.model.extra_head.weight, weights["decoder.extra_heads.0.weight"]
    )
    assert torch.equal(
        draft.model.extra_proj.bias, weights["decoder.extra_proj.0.bias"]
    )


def test_parse_mtp_requires_all_integrated_auxiliary_weights(tmp_path: Path) -> None:
    draft = ParseAuxMTPModel.__new__(ParseAuxMTPModel)
    nn.Module.__init__(draft)
    draft.model = _ParseAuxMTPInner(vocab_size=4, hidden_size=3)
    save_file(
        {"decoder.extra_heads.0.bias": torch.zeros(3)},
        tmp_path / "auxiliary_prediction_heads.safetensors.extra",
    )

    with pytest.raises(RuntimeError, match="Missing Parse 3.5 MTP tensors"):
        draft.load_weights(
            [
                ("decoder.embed_tokens.weight", torch.zeros(4, 3)),
                ("decoder.extra_heads.0.bias", torch.zeros(3)),
            ]
        )


def test_parse_mtp_rejects_wrong_integrated_tensor_shape() -> None:
    draft = ParseAuxMTPModel.__new__(ParseAuxMTPModel)
    nn.Module.__init__(draft)
    draft.model = _ParseAuxMTPInner(vocab_size=4, hidden_size=3)

    with pytest.raises(ValueError, match="decoder.extra_heads.0.weight has shape"):
        draft.load_weights([("decoder.extra_heads.0.weight", torch.zeros(2, 3))])


def test_parse_decoder_exposes_head_without_registering_duplicate_module() -> None:
    decoder = MBartDecoderNoPos.__new__(MBartDecoderNoPos)
    nn.Module.__init__(decoder)
    decoder.embed_tokens = nn.Embedding(4, 3)
    head = nn.Linear(3, 4)
    decoder._lm_head_ref = weakref.ref(head)

    assert decoder.model is decoder
    assert decoder.lm_head is head
    assert "lm_head" not in dict(decoder.named_modules())
