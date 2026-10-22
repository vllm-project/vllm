# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import SpeculativeConfig
from vllm.v1.spec_decode.registry import (
    SpeculativeMethodSpec,
    register_speculative_method,
    unregister_speculative_method,
)
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    requires_aux_hidden_states,
)

NAME = "test_plugin_method"


def configure(config: SpeculativeConfig) -> None:
    config.num_speculative_tokens = config.method_config["width"] - 1
    config.num_speculative_tokens_per_batch_size = [(1, 4, 3), (5, 64, 1)]


@pytest.fixture
def registered():
    spec = SpeculativeMethodSpec(
        "collections:Counter",
        configure=configure,
        uses_aux_hidden_states=True,
    )
    register_speculative_method(NAME, spec)
    try:
        yield spec
    finally:
        unregister_speculative_method(NAME)


def test_plugin_configures_budget_and_hidden_state_requirements(registered):
    config = SpeculativeConfig(method=NAME, method_config={"width": 4})
    assert config.num_speculative_tokens == 3
    assert config.num_speculative_tokens_per_batch_size == [(1, 4, 3), (5, 64, 1)]
    assert config.registered_method() == registered
    assert config.uses_draft_kv_cache()
    assert requires_aux_hidden_states(config)


def test_plugin_options_are_part_of_compilation_cache_key(registered):
    first = SpeculativeConfig(method=NAME, method_config={"width": 4, "rank": 128})
    same = SpeculativeConfig(method=NAME, method_config={"rank": 128, "width": 4})
    other = SpeculativeConfig(method=NAME, method_config={"width": 4, "rank": 256})
    assert first.compute_hash() == same.compute_hash()
    assert first.compute_hash() != other.compute_hash()


def test_unknown_method_is_rejected_before_model_loading():
    with pytest.raises(ValueError, match="Unknown speculative method"):
        SpeculativeConfig(method="missing_plugin", num_speculative_tokens=3)


@pytest.mark.parametrize("width", [-1, 0, 1])
def test_plugin_budget_is_validated_after_configuration(registered, width):
    with pytest.raises(ValueError, match="greater than zero"):
        SpeculativeConfig(method=NAME, method_config={"width": width})


def test_builtin_method_still_uses_builtin_configuration(registered):
    config = SpeculativeConfig(method="ngram", num_speculative_tokens=3)
    assert config.registered_method() is None
    assert config.num_speculative_tokens == 3
    assert config.method_config == {}
