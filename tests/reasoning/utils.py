# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser
from vllm.tokenizers.mistral import MistralTokenizer


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
        assert delta.reasoning == delta.reasoning_content, (
            "reasoning_content should be present for backwards compatibility"
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
    assert isinstance(reasoning_parser.model_tokenizer, MistralTokenizer), type(
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
    for delta in model_deltas:
        token_delta = [
            reasoning_parser.vocab.get(token)
            for token in reasoning_parser.model_tokenizer.tokenize(delta)
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
    assert isinstance(reasoning_parser.model_tokenizer, MistralTokenizer), type(
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
