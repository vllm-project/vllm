# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Thinking-budget enablement tests for MuseGlimmer.

MuseGlimmer has no `<think>`/`</think>` pair: reasoning is a channel, opened by
naming `self` as the recipient (` to=self<|message|>`) and closed by `<|eom|>`.
Without declared boundary strings, `ReasoningConfig.initialize_token_ids`
returns early, `reasoning_config.enabled` stays False, and any request that
sets `thinking_token_budget` is rejected with HTTP 400.

A stub tokenizer keeps these checkpoint-free; the framing markers are the
special tokens they are in the real vocabulary.
"""

import pytest

from vllm.reasoning.muse_glimmer_reasoning_parser import MuseGlimmerReasoningParser

_SPECIALS = ("<|message|>", "<|eom|>", "<|eot|>", "<|start|>")


class _StubTokenizer:
    """Longest-match tokenizer over the framing markers plus single characters."""

    def __init__(self, specials=_SPECIALS):
        pieces = list(specials) + [chr(c) for c in range(32, 127)]
        self._id_to_text = pieces
        self._text_to_id = {text: i for i, text in enumerate(pieces)}
        self._ordered = sorted(pieces, key=len, reverse=True)

    def get_vocab(self):
        return dict(self._text_to_id)

    def encode(self, text, add_special_tokens=False):
        ids = []
        pos = 0
        while pos < len(text):
            for piece in self._ordered:
                if text.startswith(piece, pos):
                    ids.append(self._text_to_id[piece])
                    pos += len(piece)
                    break
            else:
                raise AssertionError(f"unencodable text at {pos}: {text[pos:]!r}")
        return ids

    def decode(self, token_ids, **kwargs):
        return "".join(self._id_to_text[i] for i in token_ids)


@pytest.fixture
def tok():
    return _StubTokenizer()


@pytest.fixture
def parser(tok):
    return MuseGlimmerReasoningParser(tok)


def test_boundary_strings_are_what_the_model_generates(parser):
    # The leading space matters: the generation prompt ends after
    # `<|start|>assistant`, so the model emits ` to`, a different token from
    # `to`, and the budget matcher compares token ids by exact slice.
    assert parser.reasoning_start_str == " to=self<|message|>"
    # The end string is the closing marker only. The full transition
    # `<|eom|><|start|>assistant to=user<|message|>` would decide the next
    # recipient, which on a tool turn belongs to the model.
    assert parser.reasoning_end_str == "<|eom|>"


def test_reasoning_config_enables_the_budget(tok, monkeypatch):
    """`thinking_token_budget` must be enabled, with forced == natural end."""
    from vllm.config import reasoning as reasoning_module
    from vllm.config.reasoning import ReasoningConfig

    monkeypatch.setattr(
        reasoning_module, "cached_tokenizer_from_config", lambda model_config: tok
    )
    config = ReasoningConfig(reasoning_parser="muse_glimmer")
    config.initialize_token_ids(model_config=None)

    assert config.enabled, "a thinking_token_budget request would get HTTP 400"
    assert config.reasoning_start_token_ids == tok.encode(" to=self<|message|>")
    assert config.reasoning_end_token_ids == tok.encode("<|eom|>")
    assert config.natural_reasoning_end_token_ids == tok.encode("<|eom|>")


def test_cli_reasoning_end_str_still_overrides(tok, monkeypatch):
    """A user-provided end string must win over the parser's declaration."""
    from vllm.config import reasoning as reasoning_module
    from vllm.config.reasoning import ReasoningConfig

    monkeypatch.setattr(
        reasoning_module, "cached_tokenizer_from_config", lambda model_config: tok
    )
    config = ReasoningConfig(
        reasoning_parser="muse_glimmer",
        reasoning_start_str=" to=self<|message|>",
        reasoning_end_str="<|eom|><|start|>assistant to=user<|message|>",
    )
    config.initialize_token_ids(model_config=None)

    assert config.enabled
    assert config.reasoning_end_token_ids == tok.encode(
        "<|eom|><|start|>assistant to=user<|message|>"
    )
    # The parser's own declaration remains the natural end.
    assert config.natural_reasoning_end_token_ids == tok.encode("<|eom|>")
