# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.reasoning import ReasoningParser
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer


class StreamingReasoningReconstructor:

    def __init__(self):
        self.reasoning_content = None
        self.other_content = None

    def append_delta(self, delta: DeltaMessage):
        # content and the reasoning content should not be present
        # at the same time
        assert delta.content is None or delta.reasoning_content is None, (
            "Both content and reasoning content are present in the "
            "delta message")
        if delta.content is not None:
            if self.other_content is None:
                self.other_content = delta.content
            else:
                self.other_content += delta.content
        else:
            if self.reasoning_content is None:
                self.reasoning_content = delta.reasoning_content
            else:
                self.reasoning_content += delta.reasoning_content


def run_reasoning_extraction(
    reasoning_parser: ReasoningParser,
    model_output: Union[list[str], list[int]],
    request: Union[ChatCompletionRequest, None] = None,
    streaming: bool = False,
) -> tuple[Optional[str], Optional[str]]:
    if streaming:
        reconstructor = run_reasoning_extraction_streaming(
            reasoning_parser,
            model_output,
            request,
        )
        return (
            reconstructor.reasoning_content,
            reconstructor.other_content or None,
        )
    else:
        reasoning, content = run_reasoning_extraction_nonstreaming(
            reasoning_parser, model_output, request)
        return reasoning, content


def run_reasoning_extraction_nonstreaming(
    reasoning_parser: ReasoningParser,
    model_output: Union[list[str], list[int]],
    request: Union[ChatCompletionRequest, None] = None,
) -> tuple[Optional[str], Optional[str]]:
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    if model_output and isinstance(model_output[0], int):
        model_output = [int(o) for o in model_output]
        assert isinstance(reasoning_parser.model_tokenizer, MistralTokenizer)
        str_output = reasoning_parser.model_tokenizer.convert_ids_to_tokens(
            model_output)
    else:
        str_output = [str(o) for o in model_output]
    return reasoning_parser.extract_reasoning_content(
        model_output=''.join(str_output), request=request)


def run_reasoning_extraction_streaming(
    reasoning_parser: ReasoningParser,
    model_deltas: Union[list[str], list[int]],
    request: Union[ChatCompletionRequest, None] = None,
) -> StreamingReasoningReconstructor:
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    reconstructor = StreamingReasoningReconstructor()
    previous_text = ""
    previous_tokens: list[int] = []
    for model_delta in model_deltas:
        if isinstance(model_delta, int):
            assert isinstance(reasoning_parser.model_tokenizer,
                              MistralTokenizer)
            token_delta = [model_delta]
            delta = reasoning_parser.model_tokenizer.convert_ids_to_tokens(
                [model_delta])[0]
        else:
            token_delta = [
                reasoning_parser.vocab.get(token) for token in
                reasoning_parser.model_tokenizer.tokenize(model_delta)
                if token in reasoning_parser.vocab
            ]
            delta = model_delta
        current_text = previous_text + delta
        current_tokens = previous_tokens + token_delta
        delta_message = reasoning_parser.extract_reasoning_content_streaming(
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
