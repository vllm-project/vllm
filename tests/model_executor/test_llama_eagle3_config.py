# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolution of the Eagle3 aux-hidden-state count across config layouts."""

import pytest
from transformers import LlamaConfig

from vllm.model_executor.models.llama_eagle3 import _resolve_num_aux_hidden_states

pytestmark = pytest.mark.skip_global_cleanup  # pure config resolution; nothing to tear down


def test_explicit_count_wins_over_layer_lists():
    # The count and the list agree, so the explicit count is taken.
    config = LlamaConfig(num_aux_hidden_states=5)
    config.eagle_config = {"eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4, 5]}
    assert _resolve_num_aux_hidden_states(config) == 5


def test_nested_eagle_config_layout():
    config = LlamaConfig()
    config.eagle_config = {"eagle_aux_hidden_state_layer_ids": [0, 5, 11, 17]}
    assert _resolve_num_aux_hidden_states(config) == 4


def test_top_level_layer_ids_layout():
    # What Speculators writes and what the speculators config loader carries
    # through: the layer list sits at the top level of the draft config.
    config = LlamaConfig(eagle_aux_hidden_state_layer_ids=[2, 8, 15, 22])
    assert _resolve_num_aux_hidden_states(config) == 4


def test_agreeing_lists_do_not_conflict():
    config = LlamaConfig(eagle_aux_hidden_state_layer_ids=[0, 1, 2])
    config.eagle_config = {"eagle_aux_hidden_state_layer_ids": [0, 1, 2]}
    assert _resolve_num_aux_hidden_states(config) == 3


def test_conflicting_lists_rejected():
    config = LlamaConfig(eagle_aux_hidden_state_layer_ids=[0, 1])
    config.eagle_config = {"eagle_aux_hidden_state_layer_ids": [0, 1, 2]}
    with pytest.raises(ValueError, match="Conflicting Eagle3"):
        _resolve_num_aux_hidden_states(config)


def test_count_disagreeing_with_list_rejected():
    config = LlamaConfig(num_aux_hidden_states=5)
    config.eagle_config = {"eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4]}
    with pytest.raises(ValueError, match="disagrees"):
        _resolve_num_aux_hidden_states(config)


def test_default_is_three():
    assert _resolve_num_aux_hidden_states(LlamaConfig()) == 3


def test_default_warns(caplog):
    with caplog.at_level("WARNING"):
        assert _resolve_num_aux_hidden_states(LlamaConfig()) == 3
    assert "assuming the historical default of 3" in caplog.text.lower() or (
        "historical default of 3" in caplog.text
    )
