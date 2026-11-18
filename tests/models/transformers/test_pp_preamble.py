# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for detecting decoders whose pre-layer code mutates `inputs_embeds`.

Later pipeline ranks only copy their received hidden states for such decoders."""

import pytest
from torch import nn
from transformers import CTRLConfig, CTRLModel, GraniteConfig, GraniteModel

from vllm.model_executor.models.transformers.utils import (
    decoder_mutates_inputs_embeds,
)

HIDDEN_SIZE = 64


def ctrl() -> tuple[nn.Module, nn.Module]:
    """Scales `inputs_embeds` in place."""
    config = CTRLConfig(n_embd=HIDDEN_SIZE, n_head=2, n_layer=1, dff=128)
    model = CTRLModel(config)
    return model, model.h[0]


def granite() -> tuple[nn.Module, nn.Module]:
    """Scales `inputs_embeds` out of place."""
    config = GraniteConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        num_attention_heads=2,
        num_hidden_layers=1,
        embedding_multiplier=12.0,
    )
    model = GraniteModel(config)
    return model, model.layers[0]


@pytest.mark.parametrize("make_model,mutates", [(ctrl, True), (granite, False)])
def test_decoder_mutates_inputs_embeds(make_model, mutates: bool):
    model, first_layer = make_model()
    probed = decoder_mutates_inputs_embeds(model, first_layer, HIDDEN_SIZE, False)
    assert probed == mutates


def test_probe_that_cannot_stop_assumes_mutation():
    """Falls back to copying when the probe never reaches the first layer."""
    model, _ = granite()
    unused_layer = nn.Linear(1, 1)
    assert decoder_mutates_inputs_embeds(model, unused_layer, HIDDEN_SIZE, False)
