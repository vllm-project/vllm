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


def make_mock_tokenizer(
    vocab: dict[str, int],
    special_tokens: list[str] | None = None,
) -> MagicMock:
    """Create a mock tokenizer with the given vocabulary.

    Args:
        vocab: Mapping of token text to token ID.
        special_tokens: Which tokens to mark as special.  When ``None``
            (the default), every key in *vocab* is treated as special —
            convenient when the vocab only contains delimiter tokens.

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
    st = special_tokens if special_tokens is not None else list(vocab.keys())
    tokenizer.all_special_tokens = st
    tokenizer.all_special_ids = [vocab[t] for t in st if t in vocab]
    return tokenizer


@pytest.fixture
def mock_request():
    req = MagicMock(spec=ChatCompletionRequest)
    req.tools = []
    req.tool_choice = "auto"
    return req
