# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for beam-search structured-output backend initialisation.

Run without GPU:
    python -m pytest \
        tests/entrypoints/generate/beam_search/test_beam_search_so_init.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

from vllm.entrypoints.beam_search_utils import (
    init_beam_search_so_backend,
    resolve_over_cap_logprobs,
)
from vllm.sampling_params import StructuredOutputsParams


def _make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.eos_token_id = 999
    tok.encode.side_effect = lambda text, **kwargs: [ord(c) for c in text]
    return tok


def _make_vllm_config(backend: str) -> MagicMock:
    config = MagicMock()
    config.model_config.is_diffusion = False
    config.model_config.get_vocab_size.return_value = 32000
    config.structured_outputs_config.backend = backend
    return config


def test_choice_uses_trie_with_auto_backend():
    """The CHOICE fast path must survive the shared validator.

    The default ``"auto"`` backend resolves to xgrammar, whose validator
    rewrites ``choice`` into an equivalent grammar in place
    (``choice -> None``, ``grammar -> EBNF``). If the request type is read
    after that mutation, the trie fast path becomes unreachable and the
    slower grammar backend is built instead -- producing identical output,
    so only a path-level assertion catches the regression.
    """
    state = init_beam_search_so_backend(
        vllm_config=_make_vllm_config("auto"),
        tokenizer=_make_tokenizer(),
        vocab_size=32000,
        structured_outputs=StructuredOutputsParams(choice=["foo", "bar"]),
    )

    assert state.trie is not None
    assert state.backend is None
    assert state.key is None
    assert state.bitmask is None


def test_resolve_over_cap_logprobs_unlimited_returns_vocab_size():
    """``max_logprobs == -1`` must request the full vocabulary.

    Only seeing every token's logprob guarantees that an allowed token is
    never missed, which is what makes over-cap beam drops impossible under
    ``--max-logprobs -1``. The top-``len(allowed)`` logprobs by probability
    can be disjoint from the allowed set, so a smaller request cannot give
    that guarantee.
    """
    assert resolve_over_cap_logprobs(-1, 32000, 8) == 32000


def test_resolve_over_cap_logprobs_bounded_by_max_logprobs():
    """A finite ceiling bounds the request above the base."""
    assert resolve_over_cap_logprobs(100, 32000, 8) == 100


def test_resolve_over_cap_logprobs_bounded_by_vocab_size():
    """The ceiling never exceeds the vocabulary size."""
    assert resolve_over_cap_logprobs(100, 10, 8) == 10


def test_resolve_over_cap_logprobs_never_below_base():
    """The base request is a floor even when the ceiling is smaller."""
    assert resolve_over_cap_logprobs(4, 32000, 8) == 8
