# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.entrypoints.openai.engine.protocol import DeltaMessage


class Parser:
    """
    This is a class that should capture the input tokenization conversion logic
    and output tokenization conversion logic.

    Tokenizer: this comes from the engine prompt when you spin up the model
    Reasoning Parser: this is passed in as a flag
    Tool calling parser: this is passed in as a flag

    but all 3 should really be doing the same thing...

    """

    def __init__(self) -> None:
        # tokenizer
        pass

    def extract_reasoning(self):
        # TODO: implement this similar to reasoning_parser.extract_reasoning
        pass

    def extract_reasoning_streaming(self) -> DeltaMessage | None:
        # TODO: implement this similar to reasoning_parser.extract_reasoning_streaming
        pass

    def extract_tool_calls(self):
        pass
