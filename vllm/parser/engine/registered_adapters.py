# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concrete adapter classes for each registered parser engine.

These are created via :func:`make_adapters` and exposed as module-level
names so that :class:`ReasoningParserManager` and
:class:`ToolParserManager` can load them lazily.
"""

from vllm.parser.deepseek_v4 import DeepSeekV4Parser
from vllm.parser.deepseek_v32 import DeepSeekV32Parser
from vllm.parser.engine.adapters import make_adapters
from vllm.parser.gemma4 import Gemma4Parser
from vllm.parser.glm47_moe import Glm47MoeParser
from vllm.parser.inkling import InklingParser
from vllm.parser.kimi_k2 import KimiK2Parser
from vllm.parser.minimax_m2 import MinimaxM2Parser
from vllm.parser.nemotron_v3 import NemotronV3Parser
from vllm.parser.qwen3 import Qwen3Parser
from vllm.parser.seed_oss import SeedOssParser

(
    DeepSeekV32ParserReasoningAdapter,
    DeepSeekV32ParserToolAdapter,
) = make_adapters(DeepSeekV32Parser)

(
    DeepSeekV4ParserReasoningAdapter,
    DeepSeekV4ParserToolAdapter,
) = make_adapters(DeepSeekV4Parser)

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

(
    SeedOssParserReasoningAdapter,
    SeedOssParserToolAdapter,
) = make_adapters(SeedOssParser)

(
    Glm47MoeParserReasoningAdapter,
    Glm47MoeParserToolAdapter,
) = make_adapters(Glm47MoeParser)

(
    KimiK2ParserReasoningAdapter,
    KimiK2ParserToolAdapter,
) = make_adapters(KimiK2Parser)

(
    InklingParserReasoningAdapter,
    InklingParserToolAdapter,
) = make_adapters(InklingParser)
