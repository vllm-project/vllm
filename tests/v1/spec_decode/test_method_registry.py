# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the out-of-tree speculative method registry and its config hooks."""

from types import SimpleNamespace

import pytest

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.v1.spec_decode import registry
from vllm.v1.spec_decode.registry import (
    SpeculativeMethodSpec,
    get_speculative_method,
    register_speculative_method,
    registered_speculative_methods,
)
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    requires_aux_hidden_states,
)

NAME = "unit_test_method"
SPEC_CLS = "tests.v1.spec_decode.test_method_registry:DummySpeculator"


class DummySpeculator:
    pass


@pytest.fixture
def registered(monkeypatch):
    """Register methods in a registry that is restored after the test."""
    monkeypatch.setattr(registry, "_REGISTRY", {})

    def _register(spec: SpeculativeMethodSpec | None = None, name: str = NAME):
        register_speculative_method(name, spec or SpeculativeMethodSpec(SPEC_CLS))

    return _register


def test_unregistered_method_is_rejected_with_the_known_names(registered):
    with pytest.raises(ValueError, match="Unknown speculative method 'nope'"):
        SpeculativeConfig(method="nope", num_speculative_tokens=2)
    registered()
    # The message lists what a plugin registered, so a typo is easy to spot.
    with pytest.raises(ValueError, match=NAME):
        SpeculativeConfig(method="nope", num_speculative_tokens=2)


def test_registered_method_is_accepted_without_a_draft_model(registered):
    registered()
    config = SpeculativeConfig(
        method=NAME, num_speculative_tokens=4, method_config={"tree_budget": 16}
    )
    assert config.method == NAME
    assert config.method_config == {"tree_budget": 16}
    assert config.registered_method() is get_speculative_method(NAME)


def test_builtin_methods_ignore_the_registry():
    config = SpeculativeConfig(method="ngram", num_speculative_tokens=3)
    assert config.registered_method() is None
    assert config.method_config == {}
    assert get_speculative_method("ngram") is None


def test_configure_hook_runs_after_post_init(registered):
    seen = {}

    def configure(config: SpeculativeConfig) -> None:
        seen["method_config"] = dict(config.method_config)
        config.num_speculative_tokens = config.method_config["budget"]
        config.num_speculative_tokens_per_batch_size = [(1, 4, 8)]

    registered(SpeculativeMethodSpec(SPEC_CLS, configure=configure))
    config = SpeculativeConfig(method=NAME, method_config={"budget": 24})
    assert seen["method_config"] == {"budget": 24}
    assert config.num_speculative_tokens == 24
    assert config.num_speculative_tokens_per_batch_size == [(1, 4, 8)]


def test_configure_is_not_called_for_other_methods(registered):
    configure_calls = []
    registered(
        SpeculativeMethodSpec(SPEC_CLS, configure=lambda c: configure_calls.append(c))
    )
    SpeculativeConfig(method="ngram", num_speculative_tokens=3)
    assert configure_calls == []


@pytest.mark.parametrize("uses_hidden", [True, False])
def test_use_eagle_follows_the_registered_trait(registered, uses_hidden):
    registered(SpeculativeMethodSpec(SPEC_CLS, uses_target_hidden_states=uses_hidden))
    config = SpeculativeConfig(method=NAME, num_speculative_tokens=2)
    assert config.use_eagle() is uses_hidden
    # The draft KV cache group is keyed on use_eagle().
    assert config.uses_draft_kv_cache() is uses_hidden
    assert config.use_eagle_block_drop() is uses_hidden


def test_aux_hidden_states_trait(registered):
    registered(SpeculativeMethodSpec(SPEC_CLS, uses_aux_hidden_states=True))
    config = SpeculativeConfig(method=NAME, num_speculative_tokens=2)
    assert requires_aux_hidden_states(config)
    del registry._REGISTRY[NAME]
    registered(SpeculativeMethodSpec(SPEC_CLS, uses_aux_hidden_states=False))
    config = SpeculativeConfig(method=NAME, num_speculative_tokens=2)
    assert not requires_aux_hidden_states(config)


def _lookahead(config: SpeculativeConfig) -> int:
    fake_vllm_config = SimpleNamespace(
        speculative_config=config,
        num_speculative_tokens=config.num_speculative_tokens,
    )
    return VllmConfig.num_lookahead_tokens.fget(fake_vllm_config)  # type: ignore[attr-defined]


def test_num_lookahead_tokens(registered):
    registered()
    config = SpeculativeConfig(method=NAME, num_speculative_tokens=5)
    assert _lookahead(config) == 5  # EAGLE-style default: K slots

    del registry._REGISTRY[NAME]
    registered(
        SpeculativeMethodSpec(
            SPEC_CLS, num_lookahead_tokens=lambda c: c.num_speculative_tokens + 3
        )
    )
    config = SpeculativeConfig(method=NAME, num_speculative_tokens=5)
    assert _lookahead(config) == 8


def test_hash_depends_on_method_config_and_aux_trait(registered):
    registered()
    base = SpeculativeConfig(method=NAME, num_speculative_tokens=2)
    other = SpeculativeConfig(
        method=NAME, num_speculative_tokens=2, method_config={"a": 1}
    )
    same = SpeculativeConfig(
        method=NAME, num_speculative_tokens=2, method_config={"a": 1}
    )
    base_hash = base.compute_hash()
    assert base_hash != other.compute_hash()
    assert other.compute_hash() == same.compute_hash()
    # Nested options hash by content, not by insertion order.
    nested_ab = SpeculativeConfig(
        method=NAME, num_speculative_tokens=2, method_config={"t": {"a": 1, "b": 2}}
    )
    nested_ba = SpeculativeConfig(
        method=NAME, num_speculative_tokens=2, method_config={"t": {"b": 2, "a": 1}}
    )
    assert nested_ab.compute_hash() == nested_ba.compute_hash()

    del registry._REGISTRY[NAME]
    registered(SpeculativeMethodSpec(SPEC_CLS, uses_aux_hidden_states=True))
    aux = SpeculativeConfig(method=NAME, num_speculative_tokens=2)
    assert aux.compute_hash() != base_hash


def test_registration_rules(registered):
    spec = SpeculativeMethodSpec(SPEC_CLS)
    registered(spec)
    assert NAME in registered_speculative_methods()
    # A plugin entry point can run twice in one process.
    register_speculative_method(NAME, spec)
    with pytest.raises(ValueError, match="already registered"):
        register_speculative_method(NAME, SpeculativeMethodSpec(SPEC_CLS + "2"))
    for builtin in ("eagle", "eagle3", "mtp", "ngram", "custom_class"):
        with pytest.raises(ValueError, match="built-in"):
            register_speculative_method(builtin, spec)
    with pytest.raises(ValueError, match="non-empty"):
        register_speculative_method("", spec)


def test_speculator_class_is_resolved_lazily(registered):
    spec = SpeculativeMethodSpec("no_such_module_xyz:Missing")
    registered(spec)  # registering does not import
    with pytest.raises(ModuleNotFoundError):
        spec.resolve_speculator_cls()
    assert SpeculativeMethodSpec(SPEC_CLS).resolve_speculator_cls() is DummySpeculator
    # The dotted form works too.
    dotted = SPEC_CLS.replace(":", ".")
    assert SpeculativeMethodSpec(dotted).resolve_speculator_cls() is DummySpeculator


def test_repr_without_a_draft_model(registered):
    registered()
    config = SpeculativeConfig(method=NAME, num_speculative_tokens=2)
    assert NAME in repr(config)
