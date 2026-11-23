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

from vllm.config.load import LoadConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.nemotron_parse3_5 import MBartDecoderNoPos
from vllm.model_executor.models.nemotron_parse3_5_mtp import (
    ParseAuxMTPModel,
    _ParseAuxMTPInner,
)


def test_parse_mtp_config_routes_only_causal_prefix_architecture() -> None:
    config = PreTrainedConfig(architectures=["RadioMBartPrefixForCausalLM"])
    config.decoder = PreTrainedConfig()
    config.decoder.causal_vision_prefix = True
    config.decoder.vocab_size = 8
    config.decoder.d_model = 4

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
        "decoder.layers.0.fc1.weight": torch.ones(3, 3),
        "encoder.conv1.weight": torch.ones(3, 3, 1),
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
    for destination, source in (
        ("embed_tokens.weight", "decoder.embed_tokens.weight"),
        ("extra_head.weight", "decoder.extra_heads.0.weight"),
        ("extra_head.bias", "decoder.extra_heads.0.bias"),
        ("extra_proj.weight", "decoder.extra_proj.0.weight"),
        ("extra_proj.bias", "decoder.extra_proj.0.bias"),
    ):
        assert torch.equal(draft.model.get_parameter(destination), weights[source])


@pytest.mark.parametrize(
    ("missing_source", "missing_destination"),
    [
        ("decoder.extra_heads.0.weight", "model.extra_head.weight"),
        ("decoder.embed_tokens.weight", "model.embed_tokens.weight"),
    ],
)
def test_parse_mtp_default_loader_detects_missing_weights(
    missing_source: str, missing_destination: str
) -> None:
    draft = ParseAuxMTPModel.__new__(ParseAuxMTPModel)
    nn.Module.__init__(draft)
    draft.model = _ParseAuxMTPInner(vocab_size=4, hidden_size=3)
    weights = {
        "decoder.embed_tokens.weight": torch.zeros(4, 3),
        "decoder.extra_heads.0.weight": torch.zeros(3, 3),
        "decoder.extra_heads.0.bias": torch.zeros(3),
        "decoder.extra_proj.0.weight": torch.zeros(3, 3),
        "decoder.extra_proj.0.bias": torch.zeros(3),
    }
    del weights[missing_source]
    loaded = draft.load_weights(weights.items())

    loader = DefaultModelLoader(LoadConfig())
    with pytest.raises(ValueError, match="not initialized from checkpoint") as exc:
        loader.track_weights_loading(draft, loaded)
    assert missing_destination in str(exc.value)


def test_parse_decoder_exposes_head_without_registering_duplicate_module() -> None:
    decoder = MBartDecoderNoPos.__new__(MBartDecoderNoPos)
    nn.Module.__init__(decoder)
    decoder.embed_tokens = nn.Embedding(4, 3)
    head = nn.Linear(3, 4)
    decoder._lm_head_ref = weakref.ref(head)

    assert decoder.model is decoder
    assert decoder.lm_head is head
    assert "lm_head" not in dict(decoder.named_modules())


def test_parse_mtp_uses_replaced_scaled_embedding_in_auxiliary_prediction() -> None:
    class ScaledEmbedding(nn.Embedding):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return super().forward(input_ids) * 2

    model = _ParseAuxMTPInner(vocab_size=4, hidden_size=3)
    shared_embedding = ScaledEmbedding(4, 3)
    model.embed_tokens = shared_embedding
    with torch.no_grad():
        shared_embedding.weight.fill_(1)
        model.extra_proj.weight.copy_(3 * torch.eye(3))
        model.extra_proj.bias.fill_(1)
        model.extra_head.weight.copy_(5 * torch.eye(3))
        model.extra_head.bias.fill_(7)

    actual = model(torch.tensor([1]), torch.tensor([[1.0, 2.0, 3.0]]))

    # The trained formula is head(hidden + proj(scaled_embedding(token))).
    torch.testing.assert_close(actual, torch.tensor([[47.0, 52.0, 57.0]]))
