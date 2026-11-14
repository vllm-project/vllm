# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from collections.abc import Sequence
from typing import Any, cast

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.reasoning import ReasoningParser
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.v1.token_buffer import TokenBuffer, TokenIdsView
from vllm.v1.utils import ConstantList


class StreamingReasoningReconstructor:
    def __init__(self):
        self.reasoning = None
        self.other_content = None

    def append_delta(self, delta: DeltaMessage):
        # content and the reasoning content should not be present
        # at the same time
        assert delta.content is None or delta.reasoning is None, (
            "Both content and reasoning content are present in the delta message"
        )
        if delta.content is not None:
            if self.other_content is None:
                self.other_content = delta.content
            else:
                self.other_content += delta.content
        else:
            if self.reasoning is None:
                self.reasoning = delta.reasoning
            else:
                self.reasoning += delta.reasoning


def run_reasoning_extraction(
    reasoning_parser: ReasoningParser,
    model_output: list[str],
    request: ChatCompletionRequest | None = None,
    streaming: bool = False,
) -> tuple[str | None, str | None]:
    if streaming:
        reconstructor = run_reasoning_extraction_streaming(
            reasoning_parser,
            model_output,
            request,
        )
        return (
            reconstructor.reasoning,
            reconstructor.other_content or None,
        )
    else:
        reasoning, content = run_reasoning_extraction_nonstreaming(
            reasoning_parser, model_output, request
        )
        return reasoning, content


def run_reasoning_extraction_mistral(
    reasoning_parser: ReasoningParser,
    model_output: list[int],
    request: ChatCompletionRequest | None = None,
    streaming: bool = False,
) -> tuple[str | None, str | None]:
    assert is_mistral_tokenizer(reasoning_parser.model_tokenizer), type(
        reasoning_parser.model_tokenizer
    )
    if streaming:
        reconstructor = run_reasoning_extraction_streaming_mistral(
            reasoning_parser,
            model_output,
            request,
        )
        return (
            reconstructor.reasoning,
            reconstructor.other_content or None,
        )
    else:
        str_output = reasoning_parser.model_tokenizer.convert_ids_to_tokens(
            model_output
        )
        reasoning, content = run_reasoning_extraction_nonstreaming(
            reasoning_parser, str_output, request
        )
        return reasoning, content


def run_reasoning_extraction_nonstreaming(
    reasoning_parser: ReasoningParser,
    model_output: list[str],
    request: ChatCompletionRequest | None = None,
) -> tuple[str | None, str | None]:
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    return reasoning_parser.extract_reasoning(
        model_output="".join(model_output), request=request
    )


def run_reasoning_extraction_streaming(
    reasoning_parser: ReasoningParser,
    model_deltas: list[str],
    request: ChatCompletionRequest | None = None,
) -> StreamingReasoningReconstructor:
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    reconstructor = StreamingReasoningReconstructor()
    previous_text = ""
    previous_tokens: list[int] = []
    tokenizer = cast(PreTrainedTokenizerBase, reasoning_parser.model_tokenizer)
    for delta in model_deltas:
        token_delta = [
            reasoning_parser.vocab[token]
            for token in tokenizer.tokenize(delta)
            if token in reasoning_parser.vocab
        ]
        current_text = previous_text + delta
        current_tokens = previous_tokens + token_delta
        delta_message = reasoning_parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta,
            previous_tokens,
            current_tokens,
            token_delta,
        )
        if delta_message is not None:
            reconstructor.append_delta(delta_message)
        previous_text = current_text
        previous_tokens = current_tokens
    return reconstructor


def run_reasoning_extraction_streaming_mistral(
    reasoning_parser: ReasoningParser,
    model_deltas: list[int],
    request: ChatCompletionRequest | None = None,
) -> StreamingReasoningReconstructor:
    assert is_mistral_tokenizer(reasoning_parser.model_tokenizer), type(
        reasoning_parser.model_tokenizer
    )
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    reconstructor = StreamingReasoningReconstructor()
    previous_text = ""
    previous_tokens: list[int] = []
    for model_delta in model_deltas:
        token_delta = [model_delta]
        delta = reasoning_parser.model_tokenizer.convert_ids_to_tokens([model_delta])[0]
        current_text = previous_text + delta
        current_tokens = previous_tokens + token_delta
        delta_message = reasoning_parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta,
            previous_tokens,
            current_tokens,
            token_delta,
        )
        if delta_message is not None:
            reconstructor.append_delta(delta_message)
        previous_text = current_text
        previous_tokens = current_tokens
    return reconstructor


class _UnreadableIds(Sequence[int]):
    """Token ids that fail when read."""

    def __len__(self) -> int:
        raise AssertionError("input_ids was read")

    def __getitem__(self, index):
        raise AssertionError("input_ids was read")


def check_streaming_end_matches_default(
    parser: Any,
    alphabet: list[int],
    hand_cases: list[tuple[list[int], list[int], bool]],
    num_random_cases: int = 2000,
) -> None:
    """Check a parser's is_reasoning_end_streaming against the base default,
    which re-runs is_reasoning_end on the whole sequence.

    Only cases where reasoning has not ended before the delta count, since
    callers stop asking once it has. A delta without the end token must not
    read input_ids at all. ``hand_cases`` are (prefix, delta, expected).
    """
    rng = random.Random(0)
    cases: list[tuple[list[int], list[int], bool | None]] = list(hand_cases)
    for _ in range(num_random_cases):
        ids = [rng.choice(alphabet) for _ in range(rng.randint(1, 12))]
        num_delta = rng.randint(1, min(5, len(ids)))
        cases.append((ids[:-num_delta], ids[-num_delta:], None))

    num_random_checked = 0
    for prefix, delta, expected in cases:
        if parser.is_reasoning_end(prefix):
            assert expected is None, f"hand case {prefix=} has already ended"
            continue
        ids = prefix + delta
        default = ReasoningParser.is_reasoning_end_streaming(parser, ids, delta)
        if expected is not None:
            assert default == expected, (prefix, delta)
        for input_ids in (
            ids,
            ConstantList(TokenBuffer(ids)),
            TokenIdsView(TokenBuffer(prefix), delta),
        ):
            result = parser.is_reasoning_end_streaming(input_ids, delta)
            assert result == default, (prefix, delta, type(input_ids))
        if parser.end_token_id not in delta:
            assert not parser.is_reasoning_end_streaming(_UnreadableIds(), delta)
        num_random_checked += expected is None
    # The prefix filter must leave a meaningful number of cases.
    assert num_random_checked > num_random_cases // 4
