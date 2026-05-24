# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Gumiho speculative decoding drafter.

These tests target the small, pure-Python pieces of the Gumiho integration:
config wrapping, MLP-head token generation, OOB-head filtering, and the two
hook methods exposed by ``GumihoProposer``. End-to-end correctness (target +
draft model + verifier + rejection sampler) is covered by the broader
spec-decode integration suite, not here.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from transformers import LlamaConfig

from vllm.transformers_utils.configs.gumiho import GumihoConfig
from vllm.v1.spec_decode.gumiho import GumihoProposer

# ---------------------------------------------------------------------------
# GumihoConfig
# ---------------------------------------------------------------------------


def _make_llama_config() -> LlamaConfig:
    return LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        vocab_size=128,
    )


def test_gumiho_config_wraps_llama():
    """GumihoConfig should keep model_type='gumiho' and mirror Llama attrs."""
    cfg = GumihoConfig(_make_llama_config())

    assert cfg.model_type == "gumiho"
    assert cfg.architectures == ["GumihoLlamaForCausalLM"]
    # Backbone attributes should be mirrored on the outer config so vLLM's
    # standard model loading paths can read e.g. hidden_size off the top.
    assert cfg.hidden_size == 16
    assert cfg.vocab_size == 128
    assert cfg.truncated_vocab_size == 128
    # The original backbone config is preserved under ``.model`` for callers
    # that need it (e.g. EAGLE-style draft model loaders).
    assert cfg.model.model_type == "llama"


def test_gumiho_config_respects_existing_gumiho_architecture():
    """If the backbone already has a Gumiho* architecture, don't double-prefix."""
    llama = _make_llama_config()
    llama.architectures = ["GumihoLlamaForCausalLM"]

    cfg = GumihoConfig(llama)

    assert cfg.architectures == ["GumihoLlamaForCausalLM"]


def test_gumiho_config_truncated_vocab_override():
    cfg = GumihoConfig(_make_llama_config(), truncated_vocab_size=64)
    assert cfg.truncated_vocab_size == 64


def test_gumiho_config_falls_back_to_gumiho_model_for_unknown_backbone():
    """Backbones that are neither Llama nor pre-Gumiho map to ``GumihoModel``."""
    from transformers import PretrainedConfig

    backbone = PretrainedConfig()  # generic PretrainedConfig, no architectures
    backbone.architectures = None
    backbone.model_type = "unknown"

    cfg = GumihoConfig(backbone)
    assert cfg.architectures == ["GumihoModel"]


# ---------------------------------------------------------------------------
# GumihoLlamaForCausalLM.generate_mlp_draft_token_ids / _has_configured_mlp_head
#
# We build the model class manually (without ``__init__``) so we don't have to
# stand up a full VllmConfig + import all the heavy weight loaders that the
# constructor would otherwise need.
# ---------------------------------------------------------------------------


class _FuseInputsStub:
    """Stub for ``GumihoLlamaModel.fuse_inputs`` used by the MLP path."""

    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size

    def fuse_inputs(
        self, input_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # ``fuse_inputs`` returns a [batch, hidden_size] tensor; we don't care
        # about the actual values for these tests, only the shape contract.
        return torch.zeros(input_ids.shape[0], self.hidden_size)


def _make_gumiho_model(num_mlp_heads: int, hidden_size: int = 8, vocab_size: int = 16):
    """Construct a bare GumihoLlamaForCausalLM-like object for unit testing."""
    from vllm.model_executor.models.gumiho import (
        GumihoLlamaForCausalLM,
        GumihoNoResBlock,
        GumihoResBlock,
    )

    # Bypass ``GumihoLlamaForCausalLM.__init__`` (which would touch vLLM's
    # distributed init paths) and assemble only the attributes the
    # unit-under-test reads. We still need to invoke ``nn.Module.__init__``
    # so the underlying ``_parameters``/``_modules`` dicts exist.
    model = GumihoLlamaForCausalLM.__new__(GumihoLlamaForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=vocab_size, hidden_size=hidden_size)
    model.model = _FuseInputsStub(hidden_size)
    model.mlp = torch.nn.ModuleList(
        [
            torch.nn.Sequential(
                GumihoNoResBlock(hidden_size * 2, hidden_size),
                GumihoResBlock(hidden_size, hidden_size),
            )
            for _ in range(num_mlp_heads)
        ]
    )

    # ``compute_logits`` is stubbed to a deterministic argmax target so the
    # assertions below can pin the expected token ids.
    def _compute_logits(hidden_states: torch.Tensor) -> torch.Tensor:
        batch = hidden_states.shape[0]
        logits = torch.full((batch, vocab_size), -100.0)
        logits[:, 7] = 100.0
        return logits

    model.compute_logits = _compute_logits
    return model


def test_generate_mlp_draft_token_ids_shape():
    """MLP heads produce a ``[batch, num_tokens]`` int tensor."""
    model = _make_gumiho_model(num_mlp_heads=3)
    batch = 4
    hidden = 8
    draft_token_ids = [
        torch.zeros(batch, dtype=torch.long),
        torch.zeros(batch, dtype=torch.long),
    ]
    draft_hidden_states = [
        torch.randn(batch, hidden),
        torch.randn(batch, hidden),
    ]

    out = model.generate_mlp_draft_token_ids(
        draft_token_ids=draft_token_ids,
        draft_hidden_states=draft_hidden_states,
        num_tokens=2,
    )

    assert out.shape == (batch, 2)
    # Stubbed logits put all mass on token id 7.
    assert torch.all(out == 7)


@pytest.mark.parametrize(
    "num_tokens, draft_token_count, mlp_heads, expected_none",
    [
        (0, 2, 3, True),  # caller asked for 0 extra tokens
        (2, 2, 0, True),  # no MLP heads configured
        (2, 1, 3, True),  # insufficient draft history (need 2)
    ],
)
def test_generate_mlp_draft_token_ids_returns_none(
    num_tokens, draft_token_count, mlp_heads, expected_none
):
    model = _make_gumiho_model(num_mlp_heads=mlp_heads)
    batch = 2
    hidden = 8
    draft_token_ids = [
        torch.zeros(batch, dtype=torch.long) for _ in range(draft_token_count)
    ]
    draft_hidden_states = [torch.randn(batch, hidden) for _ in range(draft_token_count)]

    out = model.generate_mlp_draft_token_ids(
        draft_token_ids=draft_token_ids,
        draft_hidden_states=draft_hidden_states,
        num_tokens=num_tokens,
    )

    assert (out is None) == expected_none


@pytest.mark.parametrize(
    "weight_name, configured_heads, expected",
    [
        ("mlp.0.0.linear.weight", 2, True),
        ("mlp.1.0.linear.weight", 2, True),
        ("mlp.2.0.linear.weight", 2, False),
        ("mlp.10.0.linear.weight", 2, False),
        # Edge cases: malformed / non-integer head index -> filtered out.
        ("mlp.foo.0.linear.weight", 2, False),
        ("mlp.", 2, False),
    ],
)
def test_has_configured_mlp_head(weight_name, configured_heads, expected):
    """Skip checkpoint MLP heads beyond the configured num_speculative_tokens."""
    model = _make_gumiho_model(num_mlp_heads=configured_heads)
    assert model._has_configured_mlp_head(weight_name) is expected


# ---------------------------------------------------------------------------
# GumihoProposer hook behaviour
#
# We bypass the EagleProposer __init__ (which needs a full VllmConfig + a
# real distributed setup) by allocating a bare instance and only setting the
# attributes the two hooks touch.
# ---------------------------------------------------------------------------


def _make_gumiho_proposer(num_speculative_tokens: int) -> GumihoProposer:
    proposer = GumihoProposer.__new__(GumihoProposer)
    proposer.num_speculative_tokens = num_speculative_tokens
    return proposer


@pytest.mark.parametrize(
    "num_speculative_tokens, expects_list",
    [(1, False), (2, False), (3, True), (5, True)],
)
def test_init_draft_hidden_states_list(num_speculative_tokens, expects_list):
    """``_init_draft_hidden_states_list`` only activates when MLP heads are needed."""
    proposer = _make_gumiho_proposer(num_speculative_tokens)
    sample = torch.randn(2, 8)

    out = proposer._init_draft_hidden_states_list(sample)

    if expects_list:
        assert isinstance(out, list)
        assert len(out) == 1
        assert torch.equal(out[0], sample)
    else:
        assert out is None


def test_maybe_get_mlp_draft_token_ids_returns_none_when_history_insufficient():
    proposer = _make_gumiho_proposer(num_speculative_tokens=4)
    proposer.model = mock.MagicMock()

    # Only one hidden state collected so far; need two to fuse.
    assert (
        proposer._maybe_get_mlp_draft_token_ids(
            draft_token_ids_list=[torch.zeros(2, dtype=torch.long)],
            draft_hidden_states_list=[torch.zeros(2, 8)],
        )
        is None
    )


def test_maybe_get_mlp_draft_token_ids_returns_none_when_no_remaining():
    proposer = _make_gumiho_proposer(num_speculative_tokens=2)
    proposer.model = mock.MagicMock()

    # Two tokens already produced, none remaining.
    assert (
        proposer._maybe_get_mlp_draft_token_ids(
            draft_token_ids_list=[
                torch.zeros(2, dtype=torch.long),
                torch.zeros(2, dtype=torch.long),
            ],
            draft_hidden_states_list=[torch.zeros(2, 8), torch.zeros(2, 8)],
        )
        is None
    )


def test_maybe_get_mlp_draft_token_ids_returns_none_when_model_lacks_method():
    proposer = _make_gumiho_proposer(num_speculative_tokens=4)
    # A bare object with no ``generate_mlp_draft_token_ids`` attribute.
    proposer.model = SimpleNamespace()

    assert (
        proposer._maybe_get_mlp_draft_token_ids(
            draft_token_ids_list=[
                torch.zeros(2, dtype=torch.long),
                torch.zeros(2, dtype=torch.long),
            ],
            draft_hidden_states_list=[torch.zeros(2, 8), torch.zeros(2, 8)],
        )
        is None
    )


def test_maybe_get_mlp_draft_token_ids_calls_model_with_expected_args():
    """When all preconditions hold, the proposer should call the model with
    the first two draft tokens / hidden states and forward the result."""
    proposer = _make_gumiho_proposer(num_speculative_tokens=4)
    expected = torch.zeros(2, 2, dtype=torch.long)

    generate_mock = mock.MagicMock(return_value=expected)
    proposer.model = SimpleNamespace(generate_mlp_draft_token_ids=generate_mock)

    draft_token_ids_list = [
        torch.full((2,), 1, dtype=torch.long),
        torch.full((2,), 2, dtype=torch.long),
    ]
    draft_hidden_states_list = [
        torch.zeros(2, 8),
        torch.ones(2, 8),
    ]

    out = proposer._maybe_get_mlp_draft_token_ids(
        draft_token_ids_list, draft_hidden_states_list
    )

    assert torch.equal(out, expected)
    generate_mock.assert_called_once()
    call_kwargs = generate_mock.call_args.kwargs
    assert call_kwargs["num_tokens"] == 2  # 4 - 2 already produced
    assert call_kwargs["draft_token_ids"] == draft_token_ids_list[:2]
    assert call_kwargs["draft_hidden_states"] == draft_hidden_states_list[:2]


def test_maybe_get_mlp_draft_token_ids_returns_none_on_wrong_shape():
    """If the model returns a tensor whose 2nd dim doesn't match the number of
    remaining tokens, fall back to the sequential drafting path."""
    proposer = _make_gumiho_proposer(num_speculative_tokens=4)
    wrong_shape = torch.zeros(2, 1, dtype=torch.long)  # expected (2, 2)
    proposer.model = SimpleNamespace(
        generate_mlp_draft_token_ids=mock.MagicMock(return_value=wrong_shape)
    )

    out = proposer._maybe_get_mlp_draft_token_ids(
        draft_token_ids_list=[
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
        ],
        draft_hidden_states_list=[torch.zeros(2, 8), torch.zeros(2, 8)],
    )

    assert out is None


# ---------------------------------------------------------------------------
# SpeculativeConfig: ensure gumiho is wired into use_eagle() and method auto-detect
# ---------------------------------------------------------------------------


def test_speculative_config_recognises_gumiho_method():
    """``use_eagle()`` should return True for gumiho so the V1 GPU runner
    re-uses the EAGLE-style code paths (padded drafter, KV slot mapping, etc.)."""
    from vllm.config import SpeculativeConfig

    cfg = SpeculativeConfig.__new__(SpeculativeConfig)
    cfg.method = "gumiho"
    assert cfg.use_eagle() is True

    cfg.method = "ngram"
    assert cfg.use_eagle() is False
