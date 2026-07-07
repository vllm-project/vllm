# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser, ReasoningParserManager


class MultiTokenThinkTokenizer:
    """Small tokenizer where think markers are deliberately multi-token."""

    _piece_to_id = {
        "<": 1,
        "</": 2,
        "think": 3,
        ">": 4,
    }
    _id_to_piece = {token_id: piece for piece, token_id in _piece_to_id.items()}

    def get_vocab(self) -> dict[str, int]:
        return dict(self._piece_to_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        token_ids: list[int] = []
        idx = 0
        next_id = max(self._id_to_piece) + 1
        while idx < len(text):
            if text.startswith("</", idx):
                piece = "</"
                idx += len(piece)
            elif text.startswith("think", idx):
                piece = "think"
                idx += len(piece)
            else:
                piece = text[idx]
                idx += 1

            token_id = self._piece_to_id.get(piece)
            if token_id is None:
                token_id = next_id + ord(piece)
                self._piece_to_id[piece] = token_id
                self._id_to_piece[token_id] = piece
            token_ids.append(token_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self._id_to_piece[token_id] for token_id in token_ids)


@pytest.fixture()
def tokenizer():
    return MultiTokenThinkTokenizer()


@pytest.fixture()
def parser(tokenizer) -> ReasoningParser:
    parser_cls = ReasoningParserManager.get_reasoning_parser("rwkv")
    return parser_cls(tokenizer, chat_template_kwargs={"enable_thinking": True})



def _extract_streaming(
    parser: ReasoningParser,
    tokenizer: MultiTokenThinkTokenizer,
    deltas: list[str],
) -> tuple[str | None, str | None]:
    reasoning = ""
    content = ""
    previous_text = ""
    previous_token_ids: list[int] = []

    for delta in deltas:
        delta_ids = tokenizer.encode(delta)
        current_text = previous_text + delta
        current_token_ids = previous_token_ids + delta_ids
        delta_message = parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta,
            previous_token_ids,
            current_token_ids,
            delta_ids,
        )
        if isinstance(delta_message, DeltaMessage):
            reasoning += delta_message.reasoning or ""
            content += delta_message.content or ""
        previous_text = current_text
        previous_token_ids = current_token_ids

    return reasoning or None, content or None



def test_rwkv_parser_is_registered(tokenizer):
    parser_cls = ReasoningParserManager.get_reasoning_parser("rwkv")
    parser = parser_cls(tokenizer)

    assert parser.start_token_ids == [1, 3, 4]
    assert parser.end_token_ids == [2, 3, 4]
    assert "<think>" not in tokenizer.get_vocab()
    assert "</think>" not in tokenizer.get_vocab()


@pytest.mark.parametrize(
    ("model_output", "expected_reasoning", "expected_content"),
    [
        ("reason</think>answer", "reason", "answer"),
        ("<think>reason</think>answer", "reason", "answer"),
        ("<think>reason</think>", "reason", None),
        ("reason", "reason", None),
    ],
)
def test_rwkv_extract_reasoning_when_thinking_enabled(
    parser: ReasoningParser,
    model_output: str,
    expected_reasoning: str | None,
    expected_content: str | None,
):
    reasoning, content = parser.extract_reasoning(model_output, request=None)

    assert reasoning == expected_reasoning
    assert content == expected_content



def test_rwkv_extract_reasoning_without_forced_thinking(tokenizer):
    parser_cls = ReasoningParserManager.get_reasoning_parser("rwkv")
    parser = parser_cls(tokenizer)

    assert parser.extract_reasoning("plain answer", request=None) == (
        None,
        "plain answer",
    )
    assert parser.extract_reasoning("answer</think>", request=None) == (
        "answer",
        None,
    )
    assert parser.extract_reasoning("<think>reason</think>answer", request=None) == (
        "reason",
        "answer",
    )



def test_rwkv_token_sequence_boundaries(parser: ReasoningParser, tokenizer):
    open_prompt = tokenizer.encode("<think>\n")
    closed_prompt = tokenizer.encode("<think>\n\n</think>\n\n")
    output = tokenizer.encode("reason</think>answer")

    assert not parser.is_reasoning_end(open_prompt)
    assert parser.is_reasoning_end(closed_prompt)
    assert parser.extract_content_ids(output) == tokenizer.encode("answer")
    assert parser.count_reasoning_tokens(output) == len(tokenizer.encode("reason"))



def test_rwkv_streaming_splits_multi_token_end_marker(parser, tokenizer):
    reasoning, content = _extract_streaming(
        parser,
        tokenizer,
        ["reason", "</", "think", ">", "answer"],
    )

    assert reasoning == "reason"
    assert content == "answer"



def test_rwkv_streaming_splits_multi_token_start_marker(tokenizer):
    parser_cls = ReasoningParserManager.get_reasoning_parser("rwkv")
    parser = parser_cls(tokenizer)

    reasoning, content = _extract_streaming(
        parser,
        tokenizer,
        ["<", "think", ">", "reason", "</think>", "answer"],
    )

    assert reasoning == "reason"
    assert content == "answer"



def test_rwkv_count_reasoning_tokens_requires_thinking_or_start(tokenizer):
    parser_cls = ReasoningParserManager.get_reasoning_parser("rwkv")
    parser = parser_cls(tokenizer)

    assert parser.count_reasoning_tokens(tokenizer.encode("plain answer")) == 0
    assert parser.count_reasoning_tokens(
        tokenizer.encode("<think>reason</think>answer")
    ) == len(tokenizer.encode("reason"))
