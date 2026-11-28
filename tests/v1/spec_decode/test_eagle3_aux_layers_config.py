# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)


def _spec_config(hf_config):
    return SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=hf_config))


def test_top_level_eagle_aux_ids():
    hf = SimpleNamespace(eagle_aux_hidden_state_layer_ids=[2, 14, 27])
    assert get_eagle3_aux_layers_from_config(_spec_config(hf)) == (2, 14, 27)


def test_nested_eagle_config_ids():
    hf = SimpleNamespace(eagle_config={"eagle_aux_hidden_state_layer_ids": [2, 14, 27]})
    assert get_eagle3_aux_layers_from_config(_spec_config(hf)) == (2, 14, 27)


def test_nested_eagle_config_layer_ids():
    hf = SimpleNamespace(eagle_config={"layer_ids": [2, 14, 27]})
    assert get_eagle3_aux_layers_from_config(_spec_config(hf)) == (2, 14, 27)


def test_missing_ids_returns_none():
    hf = SimpleNamespace(eagle_config={"use_aux_hidden_state": True})
    assert get_eagle3_aux_layers_from_config(_spec_config(hf)) is None


def test_none_spec_config_returns_none():
    assert get_eagle3_aux_layers_from_config(None) is None
