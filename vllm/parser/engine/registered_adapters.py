# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concrete adapter classes for each registered parser engine.

These are created via :func:`make_adapters` and exposed as module-level
names so that :class:`ReasoningParserManager` and
:class:`ToolParserManager` can load them lazily.
"""

from vllm.parser.engine.adapters import make_adapters
from vllm.parser.gemma4 import Gemma4Parser
from vllm.parser.minimax_m2 import MinimaxM2Parser
from vllm.parser.nemotron_v3 import NemotronV3Parser
from vllm.parser.qwen3 import Qwen3Parser

(
    MinimaxM2ParserReasoningAdapter,
    MinimaxM2ParserToolAdapter,
) = make_adapters(MinimaxM2Parser)

(
    Gemma4ParserReasoningAdapter,
    Gemma4ParserToolAdapter,
) = make_adapters(Gemma4Parser)

(
    NemotronV3ParserReasoningAdapter,
    NemotronV3ParserToolAdapter,
) = make_adapters(NemotronV3Parser)

(
    Qwen3ParserReasoningAdapter,
    Qwen3ParserToolAdapter,
) = make_adapters(Qwen3Parser)
