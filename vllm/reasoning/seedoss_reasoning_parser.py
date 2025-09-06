# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


@ReasoningParserManager.register_module("seed_oss")
class SeedOSSReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for SeedOSS model.

    The SeedOSS model uses <seed:think>...</seed:think> tokens to 
    denote reasoning content text. This parser extracts 
    the reasoning content from the model output.
    Similar to DeepSeek R1, it supports cases 
    where the model doesn't generate the start token.
    """

    start_token: str = "<seed:think>"
    end_token: str = "</seed:think>"
