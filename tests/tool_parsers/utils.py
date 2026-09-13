# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from collections.abc import Iterable, Sequence

from vllm.entrypoints.generate.base.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser


class StreamingToolReconstructor:
    def __init__(self, assert_one_tool_per_delta: bool = True):
        self.tool_calls: list[ToolCall] = []
        self.other_content: str = ""
        self._assert_one_tool_per_delta = assert_one_tool_per_delta

    def append_delta(self, delta: DeltaMessage):
        if delta.content is not None:
            self.other_content += delta.content
        else:
            assert delta.tool_calls, (
                "Streaming results should have either content or tool calls (or both)"
            )
        if self._assert_one_tool_per_delta:
            # Note: This isn't strictly required by the API and may not be
            # possible to adhere to depending on the token space and number of
            # tokens per streamed response from the model, but it is required
            # by tool_use tests, so we enforce it here by default also.
            assert len(delta.tool_calls) < 2, (
                "Streaming should include only one tool call per update."
            )
        for call_delta in delta.tool_calls:
            assert call_delta.type is None or call_delta.type == "function", (
                "Streaming tool calls should only emit function calls. Got "
                f"{call_delta.type}"
            )
            current_tool_call = (
                self.tool_calls[call_delta.index]
                if call_delta.index < len(self.tool_calls)
                else None
            )
            if current_tool_call:
                assert not call_delta.function.name, (
                    "Streaming tool calls should emit the full function name "
                    f"exactly once. Got {call_delta.function.name}"
                )
                assert not call_delta.id, (
                    "Streaming tool calls must emit function id only once. Got "
                    f"{call_delta.id}"
                )
                assert call_delta.index == len(self.tool_calls) - 1, (
                    f"Incorrect index for tool delta. Got {call_delta.index}, "
                    f"expected {len(self.tool_calls) - 1}"
                )
                current_tool_call.function.arguments += call_delta.function.arguments
            else:
                assert call_delta.id is not None, (
                    "Streaming tool calls must have an id on first appearance"
                )
                assert call_delta.function.name is not None, (
                    "Streaming tool calls must have a function name on first appearance"
                )
                assert call_delta.index == len(self.tool_calls), (
                    f"Incorrect index for tool delta. Got {call_delta.index}, "
                    f"expected {len(self.tool_calls)}"
                )
                self.tool_calls.append(
                    ToolCall(
                        id=call_delta.id,
                        function=FunctionCall(
                            name=call_delta.function.name,
                            arguments=call_delta.function.arguments or "",
                        ),
                    )
                )


def run_tool_extraction(
    tool_parser: ToolParser,
    model_output: str,
    request: ChatCompletionRequest | None = None,
    streaming: bool = False,
    assert_one_tool_per_delta: bool = True,
) -> tuple[str | None, list[ToolCall]]:
    if streaming:
        reconstructor = run_tool_extraction_streaming(
            tool_parser,
            model_output,
            request,
            assert_one_tool_per_delta=assert_one_tool_per_delta,
        )
        return reconstructor.other_content or None, reconstructor.tool_calls
    else:
        extracted = run_tool_extraction_nonstreaming(tool_parser, model_output, request)
        assert extracted.tools_called == bool(extracted.tool_calls)
        return extracted.content, extracted.tool_calls


def run_tool_extraction_nonstreaming(
    tool_parser: ToolParser,
    model_output: str,
    request: ChatCompletionRequest | None = None,
) -> ExtractedToolCallInformation:
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    return tool_parser.extract_tool_calls(model_output, request)


def split_string_into_token_deltas(tokenizer: TokenizerLike, text: str) -> list[str]:
    # Split a string into a series of deltas using the provided tokenizer. Each
    # delta will be the string equivalent of a single token.
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    previously_decoded_text = ""
    deltas = []
    for i in range(1, len(token_ids) + 1):
        current_tokens = token_ids[:i]
        current_text = tokenizer.decode(current_tokens)
        new_text = current_text[len(previously_decoded_text) :]
        previously_decoded_text = current_text
        deltas.append(new_text)
    return deltas


def group_token_deltas(deltas: Sequence[str], lengths: Sequence[int]) -> list[str]:
    """Batch per-token *deltas* into chunks of the given *lengths*.

    A server batches several tokens into one streamed delta, so the text
    a parser sees is an arbitrary grouping of the token stream rather
    than one token at a time.  Any leftover tokens form a final chunk.
    """
    chunks: list[str] = []
    start = 0
    for length in lengths:
        if start >= len(deltas):
            break
        end = min(start + length, len(deltas))
        chunks.append("".join(deltas[start:end]))
        start = end
    if start < len(deltas):
        chunks.append("".join(deltas[start:]))
    return chunks


def two_chunk_groupings(n_deltas: int) -> list[list[int]]:
    """Every way to batch *n_deltas* tokens into exactly two deltas.

    Exhaustive and cheap (``n_deltas - 1`` groupings).  A tool-call tag
    that is only mishandled when it straddles one particular delta
    boundary is invisible to single-token streaming but is always caught
    here.
    """
    return [[i, n_deltas - i] for i in range(1, n_deltas)]


def random_groupings(
    n_deltas: int,
    *,
    count: int,
    seed: int,
    max_chunk: int = 4,
) -> list[list[int]]:
    """Sample *count* random batchings of *n_deltas* tokens.

    Chunk lengths are drawn uniformly from ``1..max_chunk``.  ``seed``
    makes the set reproducible so a failure can be replayed exactly.
    """
    rng = random.Random(seed)
    groupings: list[list[int]] = []
    for _ in range(count):
        lengths: list[int] = []
        remaining = n_deltas
        while remaining > 0:
            length = min(rng.randint(1, max_chunk), remaining)
            lengths.append(length)
            remaining -= length
        groupings.append(lengths)
    return groupings


def split_string_into_token_stream(
    tokenizer: TokenizerLike, text: str
) -> tuple[list[str], list[int]]:
    """Split *text* into per-token texts alongside their token ids.

    ``split_string_into_token_deltas`` returns only the texts, so a
    caller that batches tokens has to re-tokenize the joined text to
    recover ids.  Re-tokenization does not round-trip (``"<|a|>"`` as one
    string may tokenize differently than its pieces), which would make a
    batched replay feed the parser ids the model never generated.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    texts: list[str] = []
    previously_decoded_text = ""
    for i in range(1, len(token_ids) + 1):
        current_text = tokenizer.decode(token_ids[:i])
        texts.append(current_text[len(previously_decoded_text) :])
        previously_decoded_text = current_text
    return texts, token_ids


def run_tool_extraction_streaming_batched(
    tool_parser: ToolParser,
    token_texts: Sequence[str],
    token_ids: Sequence[int],
    lengths: Sequence[int],
    request: ChatCompletionRequest | None = None,
    assert_one_tool_per_delta: bool = True,
) -> StreamingToolReconstructor:
    """Stream a token sequence batched into deltas of *lengths* tokens.

    Unlike :func:`run_tool_extraction_streaming`, the real token ids are
    carried through the batching instead of being recovered by
    re-tokenizing each delta, which matches what the server passes to
    ``extract_tool_calls_streaming``.
    """
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    reconstructor = StreamingToolReconstructor(
        assert_one_tool_per_delta=assert_one_tool_per_delta
    )
    previous_text = ""
    previous_tokens: list[int] = []
    for start, end in _batch_bounds(len(token_texts), lengths):
        delta = "".join(token_texts[start:end])
        token_delta = list(token_ids[start:end])
        current_text = previous_text + delta
        current_tokens = previous_tokens + token_delta
        delta_message = tool_parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta,
            previous_tokens,
            current_tokens,
            token_delta,
            request,
        )
        if delta_message is not None:
            reconstructor.append_delta(delta_message)
        previous_text = current_text
        previous_tokens = current_tokens
    return reconstructor


def _batch_bounds(n_tokens: int, lengths: Sequence[int]) -> list[tuple[int, int]]:
    """Turn per-delta *lengths* into ``(start, end)`` pairs over the stream."""
    bounds: list[tuple[int, int]] = []
    start = 0
    for length in lengths:
        if start >= n_tokens:
            break
        end = min(start + length, n_tokens)
        bounds.append((start, end))
        start = end
    if start < n_tokens:
        bounds.append((start, n_tokens))
    return bounds


def run_tool_extraction_streaming(
    tool_parser: ToolParser,
    model_deltas: Iterable[str],
    request: ChatCompletionRequest | None = None,
    assert_one_tool_per_delta: bool = True,
) -> StreamingToolReconstructor:
    if isinstance(model_deltas, str):
        model_deltas = split_string_into_token_deltas(
            tool_parser.model_tokenizer, model_deltas
        )

    request = request or ChatCompletionRequest(messages=[], model="test-model")
    reconstructor = StreamingToolReconstructor(
        assert_one_tool_per_delta=assert_one_tool_per_delta
    )
    previous_text = ""
    previous_tokens: list[int] = []
    for delta in model_deltas:
        token_delta = [
            tool_parser.vocab.get(token)
            for token in tool_parser.model_tokenizer.tokenize(delta)
            if token in tool_parser.vocab
        ]
        current_text = previous_text + delta
        current_tokens = previous_tokens + token_delta
        delta_message = tool_parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta,
            previous_tokens,
            current_tokens,
            token_delta,
            request,
        )
        if delta_message is not None:
            reconstructor.append_delta(delta_message)
        previous_text = current_text
        previous_tokens = current_tokens
    return reconstructor
