# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concrete adapter classes for each registered parser engine.

These are created via :func:`make_adapters` and exposed as module-level
names so that :class:`ReasoningParserManager` and
:class:`ToolParserManager` can load them lazily.
"""

from vllm.parser.engine.adapters import make_adapters
from vllm.parser.qwen3 import Qwen3Parser

(
    Qwen3ParserReasoningAdapter,
    Qwen3ParserToolAdapter,
) = make_adapters(Qwen3Parser)
