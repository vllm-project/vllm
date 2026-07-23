# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import string
from collections.abc import Sequence
from unittest.mock import Mock

import pytest

from vllm.config import VllmConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.skip_global_cleanup


class MiniMaxM3Tokenizer:
    """Small tokenizer with MiniMax M3 reasoning tags as special tokens."""

    special_tokens = ("<mm:think>", "</mm:think>")

    def __init__(self):
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        for token in self.special_tokens:
            self._add_token(token)
        for char in string.printable:
            self._add_token(char)

    def _add_token(self, token: str) -> int:
        token_id = self._token_to_id.get(token)
        if token_id is None:
            token_id = len(self._token_to_id) + 1
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return token_id

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        return [self._add_token(token) for token in self.tokenize(text)]

    def decode(
        self, ids: Sequence[int] | int, skip_special_tokens: bool = False
    ) -> str:
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self._id_to_token[token_id] for token_id in ids)

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        pos = 0
        while pos < len(text):
            for special_token in self.special_tokens:
                if text.startswith(special_token, pos):
                    tokens.append(special_token)
                    pos += len(special_token)
                    break
            else:
                tokens.append(text[pos])
                pos += 1
        return tokens

    def convert_ids_to_tokens(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        return [self._id_to_token[token_id] for token_id in ids]

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self._add_token(tokens)
        return [self._add_token(token) for token in tokens]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)


class SplitMiniMaxM3Tokenizer(MiniMaxM3Tokenizer):
    """Tokenizer that exposes marker vocab entries but encodes them as text."""

    def tokenize(self, text: str) -> list[str]:
        return list(text)


class RuntimeSplitMiniMaxM3Tokenizer(MiniMaxM3Tokenizer):
    """Tokenizer whose runtime output splits markers despite atomic encodes."""

    def encode_runtime(self, text: str) -> list[int]:
        return [self._add_token(token) for token in list(text)]


class MiniMaxM3DelegatingParser(DelegatingParser):
    reasoning_parser_cls = MiniMaxM3ReasoningParser


def make_parser(
    chat_template_kwargs: dict[str, str] | None = None,
) -> tuple[MiniMaxM3ReasoningParser, MiniMaxM3Tokenizer]:
    tokenizer = MiniMaxM3Tokenizer()
    return (
        MiniMaxM3ReasoningParser(tokenizer, chat_template_kwargs=chat_template_kwargs),
        tokenizer,
    )


def run_streaming(
    parser: MiniMaxM3ReasoningParser,
    tokenizer: MiniMaxM3Tokenizer,
    chunks: list[str],
) -> tuple[str | None, str | None, list[bool]]:
    previous_text = ""
    previous_token_ids: list[int] = []
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    reasoning_end_states: list[bool] = []

    for chunk in chunks:
        encode_runtime = getattr(tokenizer, "encode_runtime", tokenizer.encode)
        delta_token_ids = encode_runtime(chunk)
        current_text = previous_text + chunk
        current_token_ids = previous_token_ids + delta_token_ids
        delta = parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
        )
        reasoning_end_states.append(
            parser.is_reasoning_end_streaming(current_token_ids, delta_token_ids)
        )

        if delta is not None:
            if delta.reasoning is not None:
                reasoning_parts.append(delta.reasoning)
            if delta.content is not None:
                content_parts.append(delta.content)

        previous_text = current_text
        previous_token_ids = current_token_ids

    return (
        "".join(reasoning_parts) or None,
        "".join(content_parts) or None,
        reasoning_end_states,
    )


def test_parser_registration():
    parser_cls = ReasoningParserManager.get_reasoning_parser("minimax_m3")

    assert parser_cls is MiniMaxM3ReasoningParser


def test_nonstreaming_extracts_explicit_reasoning_block():
    parser, _ = make_parser()
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning(
        "<mm:think>plan</mm:think>answer", request
    )

    assert reasoning == "plan"
    assert content == "answer"


def test_nonstreaming_without_start_tag_is_content():
    parser, _ = make_parser()
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning("plain answer", request)

    assert reasoning is None
    assert content == "plain answer"


def test_nonstreaming_drops_leading_end_tag():
    parser, _ = make_parser()
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning("</mm:think>answer", request)

    assert reasoning is None
    assert content == "answer"


def test_nonstreaming_non_leading_end_tag_is_content():
    parser, _ = make_parser()
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning("XXX</mm:think>YYY", request)

    assert reasoning is None
    assert content == "XXX</mm:think>YYY"


def test_nonstreaming_enabled_mode_starts_in_reasoning():
    parser, _ = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning("plan</mm:think>answer", request)

    assert reasoning == "plan"
    assert content == "answer"


def test_nonstreaming_open_reasoning_block():
    parser, _ = make_parser()
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning("<mm:think>still thinking", request)

    assert reasoning == "still thinking"
    assert content is None


def test_streaming_reasoning_tags_are_not_returned():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["<mm:think>", "plan", "</mm:think>", "answer"],
    )

    assert reasoning == "plan"
    assert content == "answer"
    assert end_states == [False, False, True, True]


@pytest.mark.parametrize(
    ("thinking_mode", "prompt_suffix", "chunks", "expected_reasoning"),
    [
        (
            "adaptive",
            "",
            ["<mm:think>", "plan", "</mm:think>", "answer"],
            "plan",
        ),
        ("adaptive", "", ["answer"], ""),
        ("enabled", "<mm:think>", ["plan", "</mm:think>", "answer"], "plan"),
        ("disabled", "</mm:think>", ["answer"], ""),
    ],
)
def test_streaming_prompt_instructions_respect_thinking_mode(
    thinking_mode, prompt_suffix, chunks, expected_reasoning
):
    tokenizer = MiniMaxM3Tokenizer()
    parser = MiniMaxM3DelegatingParser(
        tokenizer, chat_template_kwargs={"thinking_mode": thinking_mode}
    )
    request = ChatCompletionRequest(messages=[], model="test-model")
    prompt_token_ids = tokenizer.encode(
        "When thinking is enabled, wrap reasoning in "
        f"<mm:think></mm:think> tags.\nai\n{prompt_suffix}"
    )
    reasoning_parts: list[str] = []
    content_parts: list[str] = []

    for index, chunk in enumerate(chunks):
        delta = parser.parse_delta(
            chunk,
            tokenizer.encode(chunk),
            request,
            prompt_token_ids=prompt_token_ids if index == 0 else None,
            finished=index == len(chunks) - 1,
        )
        if delta is not None and delta.reasoning is not None:
            reasoning_parts.append(delta.reasoning)
        if delta is not None and delta.content is not None:
            content_parts.append(delta.content)

    assert "".join(reasoning_parts) == expected_reasoning
    assert "".join(content_parts) == "answer"


def test_streaming_boundary_can_emit_reasoning_and_content():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["<mm:think>plan</mm:think>answer"],
    )

    assert reasoning == "plan"
    assert content == "answer"
    assert end_states == [True]


def test_streaming_drops_leading_end_tag():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["</mm:think>", "answer"],
    )

    assert reasoning is None
    assert content == "answer"
    assert end_states == [True, True]


def test_streaming_non_leading_end_tag_is_content():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["XXX</mm:think>YYY"],
    )

    assert reasoning is None
    assert content == "XXX</mm:think>YYY"
    assert end_states == [True]


def test_streaming_enabled_mode_starts_in_reasoning():
    parser, tokenizer = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["plan", "</mm:think>", "answer"],
    )

    assert reasoning == "plan"
    assert content == "answer"
    assert end_states == [False, True, True]


def test_streaming_plain_content_ends_reasoning_phase():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["plain ", "answer"],
    )

    assert reasoning is None
    assert content == "plain answer"
    assert end_states == [True, True]


def test_streaming_split_marker_tokens_are_not_returned():
    tokenizer = RuntimeSplitMiniMaxM3Tokenizer()
    parser = MiniMaxM3ReasoningParser(tokenizer)

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["<mm:think>", "Reasoning", " content", "</mm:think>", "content"],
    )

    assert reasoning == "Reasoning content"
    assert content == "content"
    assert end_states == [False, False, False, True, True]


def test_streaming_split_marker_text_drives_end_state():
    tokenizer = RuntimeSplitMiniMaxM3Tokenizer()
    parser = MiniMaxM3ReasoningParser(tokenizer)
    previous_text = ""
    previous_token_ids: list[int] = []

    for chunk in ["<mm:think>", "Reasoning", " content", "</mm:think>"]:
        delta_token_ids = tokenizer.encode_runtime(chunk)
        current_text = previous_text + chunk
        current_token_ids = previous_token_ids + delta_token_ids
        parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
        )
        previous_text = current_text
        previous_token_ids = current_token_ids

    assert parser.is_reasoning_end_streaming(previous_token_ids, []) is True


def test_streaming_split_end_marker_content_ids_are_stripped():
    tokenizer = RuntimeSplitMiniMaxM3Tokenizer()
    parser = MiniMaxM3ReasoningParser(tokenizer)
    previous_text = "<mm:think>Reasoning"
    previous_token_ids = tokenizer.encode_runtime(previous_text)
    delta_text = "</mm:think>content"
    delta_token_ids = tokenizer.encode_runtime(delta_text)
    current_token_ids = previous_token_ids + delta_token_ids

    parser.extract_reasoning_streaming(
        previous_text=previous_text,
        current_text=previous_text + delta_text,
        delta_text=delta_text,
        previous_token_ids=previous_token_ids,
        current_token_ids=current_token_ids,
        delta_token_ids=delta_token_ids,
    )

    assert parser.is_reasoning_end_streaming(current_token_ids, delta_token_ids)
    assert tokenizer.decode(parser.extract_content_ids(delta_token_ids)) == "content"


def test_streaming_split_marker_tokens_enabled_mode():
    tokenizer = RuntimeSplitMiniMaxM3Tokenizer()
    parser = MiniMaxM3ReasoningParser(
        tokenizer, chat_template_kwargs={"thinking_mode": "enabled"}
    )

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["Reasoning", " content", "</mm:think>", "content"],
    )

    assert reasoning == "Reasoning content"
    assert content == "content"
    assert end_states == [False, False, True, True]


def test_streaming_split_marker_text_across_deltas():
    tokenizer = RuntimeSplitMiniMaxM3Tokenizer()
    parser = MiniMaxM3ReasoningParser(tokenizer)

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["<mm:", "think>", "Reasoning", " content", "</mm:", "think>", "content"],
    )

    assert reasoning == "Reasoning content"
    assert content == "content"
    assert end_states == [False, False, False, False, False, True, True]


def test_streaming_split_leading_end_marker_text_across_deltas():
    tokenizer = RuntimeSplitMiniMaxM3Tokenizer()
    parser = MiniMaxM3ReasoningParser(tokenizer)

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["</mm:", "think>", "content"],
    )

    assert reasoning is None
    assert content == "content"
    assert end_states == [False, True, True]


def test_token_id_helpers_with_split_marker_tokens():
    tokenizer = SplitMiniMaxM3Tokenizer()
    parser = MiniMaxM3ReasoningParser(tokenizer)
    output_ids = tokenizer.encode(
        "<mm:think>abc</mm:think>def", add_special_tokens=False
    )
    open_reasoning_ids = tokenizer.encode("<mm:think>abc", add_special_tokens=False)
    content_ids = tokenizer.encode("plain", add_special_tokens=False)

    assert parser.is_reasoning_end(output_ids)
    assert not parser.is_reasoning_end(open_reasoning_ids)
    assert not parser.is_reasoning_end(content_ids)
    assert tokenizer.decode(parser.extract_content_ids(output_ids)) == "def"
    assert parser.extract_content_ids(open_reasoning_ids) == []
    assert parser.extract_content_ids(content_ids) == content_ids
    assert parser.count_reasoning_tokens(output_ids) == len(tokenizer.encode("abc"))


def test_token_id_helpers():
    parser, tokenizer = make_parser()
    output_ids = tokenizer.encode(
        "<mm:think>abc</mm:think>def", add_special_tokens=False
    )
    open_reasoning_ids = tokenizer.encode("<mm:think>abc", add_special_tokens=False)
    content_ids = tokenizer.encode("plain", add_special_tokens=False)

    assert parser.is_reasoning_end(output_ids)
    assert not parser.is_reasoning_end(open_reasoning_ids)
    assert not parser.is_reasoning_end(content_ids)
    assert tokenizer.decode(parser.extract_content_ids(output_ids)) == "def"
    assert parser.extract_content_ids(open_reasoning_ids) == []
    assert parser.extract_content_ids(content_ids) == content_ids
    assert parser.count_reasoning_tokens(output_ids) == len(tokenizer.encode("abc"))


def test_token_id_helpers_enabled_mode():
    parser, tokenizer = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
    output_ids = tokenizer.encode("abc</mm:think>def", add_special_tokens=False)
    open_reasoning_ids = tokenizer.encode("abc", add_special_tokens=False)

    assert parser.is_reasoning_end(output_ids)
    assert not parser.is_reasoning_end(open_reasoning_ids)
    assert tokenizer.decode(parser.extract_content_ids(output_ids)) == "def"
    assert parser.extract_content_ids(open_reasoning_ids) == []
    assert parser.count_reasoning_tokens(output_ids) == len(tokenizer.encode("abc"))
    assert parser.count_reasoning_tokens(open_reasoning_ids) == len(
        tokenizer.encode("abc")
    )


@pytest.mark.parametrize(
    ("thinking_mode", "expected"),
    [
        (None, None),
        ("adaptive", None),
        ("enabled", False),
        ("disabled", True),
    ],
)
def test_prompt_reasoning_state_follows_thinking_mode(thinking_mode, expected):
    chat_template_kwargs = (
        None if thinking_mode is None else {"thinking_mode": thinking_mode}
    )
    parser, tokenizer = make_parser(chat_template_kwargs=chat_template_kwargs)
    prompt_token_ids = tokenizer.encode(
        "Instructions may contain <mm:think></mm:think> examples.\nai\n"
    )

    assert parser.is_reasoning_end_from_prompt(prompt_token_ids) is expected


def make_structured_output_manager(
    tokenizer: MiniMaxM3Tokenizer,
) -> StructuredOutputManager:
    config = Mock(spec=VllmConfig)
    config.model_config.skip_tokenizer_init = True
    config.scheduler_config.max_num_seqs = 1
    config.structured_outputs_config.enable_in_reasoning = False
    config.speculative_config = None
    manager = StructuredOutputManager(config)
    manager.reasoner_cls = MiniMaxM3ReasoningParser
    manager.tokenizer = tokenizer
    return manager


def make_structured_output_request(
    tokenizer: MiniMaxM3Tokenizer, thinking_mode: str = "adaptive"
) -> Mock:
    request = Mock(spec=Request)
    request.request_id = "test-request"
    request.use_structured_output = True
    request.prompt_token_ids = tokenizer.encode(
        "Instructions contain <mm:think></mm:think> examples.\nai\n"
    )
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request.all_token_ids = list(request.prompt_token_ids)
    request.num_computed_tokens = len(request.prompt_token_ids)
    request.num_output_placeholders = 0
    structured_request = Mock()
    structured_request.reasoning_ended = None
    structured_request.reasoning_end_token_index = None
    structured_request.deferred_grammar_start_index = None
    structured_request.reasoning_parser_kwargs = {
        "chat_template_kwargs": {"thinking_mode": thinking_mode}
    }
    structured_request.reasoner = None
    request.structured_output_request = structured_request
    return request


def append_structured_output_step(request: Mock, token_ids: list[int]) -> None:
    request.num_computed_tokens = len(request.all_token_ids)
    request.all_token_ids.extend(token_ids)


def test_adaptive_structured_output_advances_direct_content():
    tokenizer = MiniMaxM3Tokenizer()
    manager = make_structured_output_manager(tokenizer)
    request = make_structured_output_request(tokenizer)

    assert manager.should_fill_bitmask(request) is False
    assert request.structured_output_request.reasoning_ended is None

    content_ids = tokenizer.encode("{")
    append_structured_output_step(request, content_ids)

    assert manager.should_advance(request) is True
    assert request.structured_output_request.reasoning_ended is True
    assert manager.trim_reasoning_for_advance(request, content_ids) == content_ids


def test_adaptive_structured_output_replays_ambiguous_direct_prefix():
    tokenizer = RuntimeSplitMiniMaxM3Tokenizer()
    manager = make_structured_output_manager(tokenizer)
    request = make_structured_output_request(tokenizer)

    prefix_ids = tokenizer.encode_runtime("<m")
    append_structured_output_step(request, prefix_ids)

    assert manager.should_advance(request) is False
    assert request.structured_output_request.reasoning_ended is None

    content_ids = tokenizer.encode_runtime("X")
    append_structured_output_step(request, content_ids)

    assert manager.should_advance(request) is True
    advance_ids = manager.trim_reasoning_for_advance(request, content_ids)
    assert tokenizer.decode(advance_ids) == "<mX"


def test_adaptive_structured_output_tracks_split_reasoning_markers():
    tokenizer = RuntimeSplitMiniMaxM3Tokenizer()
    manager = make_structured_output_manager(tokenizer)
    request = make_structured_output_request(tokenizer)

    append_structured_output_step(request, tokenizer.encode_runtime("<mm:"))
    assert manager.should_advance(request) is False
    assert request.structured_output_request.reasoning_ended is None

    append_structured_output_step(request, tokenizer.encode_runtime("think>"))
    assert manager.should_advance(request) is False
    assert request.structured_output_request.reasoning_ended is None

    append_structured_output_step(request, tokenizer.encode_runtime("plan</mm:think>"))
    assert manager.should_advance(request) is False
    assert request.structured_output_request.reasoning_ended is True


def test_adaptive_structured_output_skips_prompt_embed_placeholders():
    tokenizer = MiniMaxM3Tokenizer()
    manager = make_structured_output_manager(tokenizer)
    request = make_structured_output_request(tokenizer)
    request.prompt_token_ids = None
    request.num_prompt_tokens = 3
    request.all_token_ids = [0] * request.num_prompt_tokens

    content_ids = tokenizer.encode("{")
    append_structured_output_step(request, content_ids)

    assert manager.should_advance(request) is True
    assert manager.trim_reasoning_for_advance(request, content_ids) == content_ids
