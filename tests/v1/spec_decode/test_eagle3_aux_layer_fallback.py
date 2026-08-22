# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from types import SimpleNamespace

import pytest

from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    set_eagle3_aux_hidden_state_layers,
)

DEFAULT_LAYERS = (2, 40, 77)
CONFIGURED_LAYERS = (1, 23, 44)


class _DummyEagle3Model:
    supports_eagle3 = True
    has_own_lm_head = False
    has_own_embed_tokens = False

    def __init__(self):
        self.aux_layers = None

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return DEFAULT_LAYERS

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.aux_layers = layers


def _spec_config(layer_ids=None):
    hf_config = SimpleNamespace(eagle_aux_hidden_state_layer_ids=layer_ids)
    return SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=hf_config))


def test_warns_when_aux_layers_fall_back_to_default(caplog_vllm):
    model = _DummyEagle3Model()

    with caplog_vllm.at_level(logging.WARNING):
        set_eagle3_aux_hidden_state_layers(model, _spec_config())

    assert model.aux_layers == DEFAULT_LAYERS
    warnings = [r for r in caplog_vllm.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "eagle_aux_hidden_state_layer_ids" in message
    assert str(DEFAULT_LAYERS) in message


def test_does_not_warn_when_aux_layers_are_configured(caplog_vllm):
    model = _DummyEagle3Model()

    with caplog_vllm.at_level(logging.WARNING):
        set_eagle3_aux_hidden_state_layers(model, _spec_config(list(CONFIGURED_LAYERS)))

    assert model.aux_layers == CONFIGURED_LAYERS
    assert not [r for r in caplog_vllm.records if r.levelno == logging.WARNING]


if __name__ == "__main__":
    pytest.main([__file__])
