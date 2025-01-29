from typing import List, Optional, Tuple, Union

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.entrypoints.openai.reasoning_parsers import ReasoningParser


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
    model_output: List[str],
    request: Union[ChatCompletionRequest, None] = None,
    streaming: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
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
    model_output: List[str],
    request: Union[ChatCompletionRequest, None] = None,
) -> Tuple[Optional[str], Optional[str]]:
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    return reasoning_parser.extract_reasoning_content(
        model_output=''.join(model_output), request=request)


def run_reasoning_extraction_streaming(
    reasoning_parser: ReasoningParser,
    model_deltas: List[str],
    request: Union[ChatCompletionRequest, None] = None,
) -> StreamingReasoningReconstructor:
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    reconstructor = StreamingReasoningReconstructor()
    previous_text = ""
    previous_tokens: List[int] = []
    for delta in model_deltas:
        token_delta = [
            reasoning_parser.vocab.get(token)
            for token in reasoning_parser.model_tokenizer.tokenize(delta)
            if token in reasoning_parser.vocab
        ]
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
