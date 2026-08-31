# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolution of the Eagle3 aux-hidden-state count across config layouts."""

from transformers import LlamaConfig

from vllm.model_executor.models.llama_eagle3 import _resolve_num_aux_hidden_states


def test_explicit_count_wins_over_layer_lists():
    config = LlamaConfig(num_aux_hidden_states=5)
    config.eagle_config = {"eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4]}
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


def test_nested_layout_beats_top_level():
    config = LlamaConfig(eagle_aux_hidden_state_layer_ids=[0, 1])
    config.eagle_config = {"eagle_aux_hidden_state_layer_ids": [0, 1, 2]}
    assert _resolve_num_aux_hidden_states(config) == 3


def test_default_is_three():
    assert _resolve_num_aux_hidden_states(LlamaConfig()) == 3
