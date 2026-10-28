# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raon-1-mini's thinking markers and XML tool calls use the Qwen3 grammar."""

from vllm.parser.qwen3 import Qwen3Parser


class Raon1MiniParser(Qwen3Parser):
    CONFIG_NAME = "raon_1"
    TURN_BOUNDARIES = frozenset(("<|im_start|>", "<|im_end|>", "<|endoftext|>"))
