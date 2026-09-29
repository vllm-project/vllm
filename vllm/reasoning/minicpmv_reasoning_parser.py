# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.parser.minicpmv import MiniCPMVParser


class MiniCPMVParserReasoningAdapter(ParserEngineReasoningAdapter):
    # Reasoning-only on purpose: this family registers no tool parser, because
    # MiniCPM-V emits the Qwen3-Coder XML syntax verbatim and `qwen3_coder`
    # already covers it. ``make_adapters`` only builds reasoning/tool pairs, so
    # the two halves are declared explicitly here.
    _parser_engine_cls = MiniCPMVParser


__all__ = ["MiniCPMVParserReasoningAdapter"]
