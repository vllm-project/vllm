# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.reasoning import ReasoningParser


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
    model_output: str,
    output_tokens: list[str],
    token_ids: list[int],
    request: Union[ChatCompletionRequest, None] = None,
    streaming: bool = False,
) -> tuple[Optional[str], Optional[str]]:
    if streaming:
        reconstructor = run_reasoning_extraction_streaming(
            reasoning_parser,
            output_tokens,
            request,
        )
        return (
            reconstructor.reasoning_content,
            reconstructor.other_content or None,
        )
    else:
        reasoning, content = run_reasoning_extraction_nonstreaming(
            reasoning_parser, model_output, token_ids, request)
        return reasoning, content


def run_reasoning_extraction_nonstreaming(
    reasoning_parser: ReasoningParser,
    model_output: str,
    token_ids: list[int],
    request: Union[ChatCompletionRequest, None] = None,
) -> tuple[Optional[str], Optional[str]]:
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    return reasoning_parser.extract_reasoning_content(
        token_ids=token_ids, model_output=model_output, request=request)


def run_reasoning_extraction_streaming(
    reasoning_parser: ReasoningParser,
    output_tokens: list[str],
    request: Union[ChatCompletionRequest, None] = None,
) -> StreamingReasoningReconstructor:
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    reconstructor = StreamingReasoningReconstructor()
    previous_text = ""
    previous_token_ids: list[int] = []
    for token in output_tokens:
        delta_token_ids = reasoning_parser.model_tokenizer.\
            convert_tokens_to_ids([token])
        delta_text = reasoning_parser.model_tokenizer.convert_tokens_to_string(
            [token])
        current_text = previous_text + delta_text
        current_token_ids = previous_token_ids + delta_token_ids
        delta_message = reasoning_parser.extract_reasoning_content_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if delta_message is not None:
            reconstructor.append_delta(delta_message)
        previous_text = current_text
        previous_token_ids = current_token_ids
    return reconstructor
