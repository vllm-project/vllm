# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tool-parser/tokenizer compatibility is validated when OnlineRenderer is
built, so an incompatible combination fails at startup instead of on every
chat request."""

from unittest.mock import MagicMock

import pytest

from vllm.renderers.online_renderer import OnlineRenderer

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _make_online_renderer(
    vocab: dict[str, int] | None,
    enable_auto_tools: bool = True,
) -> OnlineRenderer:
    tokenizer = None
    if vocab is not None:
        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = vocab
        tokenizer.supports_grammar = False

    renderer = MagicMock()
    renderer.tokenizer = tokenizer
    if tokenizer is None:
        # Mirror BaseRenderer.get_tokenizer under skip_tokenizer_init.
        renderer.get_tokenizer.side_effect = ValueError("Tokenizer not available")

    model_config = MagicMock()
    model_config.hf_config.model_type = "test"
    model_config.model = "test"

    return OnlineRenderer(
        model_config=model_config,
        renderer=renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        enable_auto_tools=enable_auto_tools,
        tool_parser="mistral",
    )


def test_incompatible_tokenizer_fails_at_init():
    with pytest.raises(
        RuntimeError,
        match="could not locate the tool call token in the tokenizer",
    ):
        _make_online_renderer({})


def test_compatible_tokenizer_succeeds():
    renderer = _make_online_renderer({"[TOOL_CALLS]": 1})

    assert renderer.parser is not None


def test_skip_tokenizer_init_skips_validation():
    _make_online_renderer(None)


def test_auto_tools_disabled_skips_validation():
    renderer = _make_online_renderer({}, enable_auto_tools=False)

    assert renderer.parser is None
