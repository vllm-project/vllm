# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser


def test_gptoss_reasoning_ended_is_true():
    parser = GptOssReasoningParser(Mock())
    assert parser.is_reasoning_end([]) is True
    assert parser.is_reasoning_end_streaming([], []) is True
