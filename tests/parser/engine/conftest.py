# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)


@pytest.fixture()
def should_do_global_cleanup_after_test() -> bool:
    return False


def make_mock_tokenizer(vocab: dict[str, int]) -> MagicMock:
    """Create a mock tokenizer with the given special-token vocabulary.

    The returned mock supports get_vocab(), encode(), and decode().
    decode() maps known token IDs back to their text and falls back to
    chr(id) for ASCII IDs or ``<id>`` for others.
    """
    id_to_text = {v: k for k, v in vocab.items()}
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.get_vocab.return_value = dict(vocab)
    tokenizer.decode.side_effect = lambda ids: "".join(
        id_to_text.get(i, chr(i) if i < 128 else f"<{i}>") for i in ids
    )
    return tokenizer


@pytest.fixture
def mock_request():
    req = MagicMock(spec=ChatCompletionRequest)
    req.tools = []
    req.tool_choice = "auto"
    return req
