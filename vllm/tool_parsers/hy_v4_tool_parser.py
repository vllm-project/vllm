# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.parser.engine.adapters import ParserEngineToolAdapter
from vllm.parser.hy_v4 import HYV4Parser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel

if TYPE_CHECKING:
    from xgrammar import StructuralTag

logger = init_logger(__name__)


class HYV4ToolParser(ParserEngineToolAdapter):
    _parser_engine_cls = HYV4Parser
    _parser_engine: HYV4Parser
    supports_required_and_named = False
    structural_tag_model = "hy_v4"

    def __init__(self, tokenizer: TokenizerLike, tools=None, **kwargs):
        super().__init__(tokenizer, tools, thinking=False, **kwargs)

    def get_structural_tag(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
        *,
        reasoning: bool = False,
        strict_level: ToolStrictLevel = ToolStrictLevel.AUTO,
    ) -> StructuralTag | None:
        """Constrain required/named calls with the checkpoint's native markers."""
        if not envs.VLLM_ENFORCE_STRICT_TOOL_CALLING:
            return None

        # Auto tool choice keeps free generation without per-token grammar work.
        tool_choice = request.tool_choice
        if (
            tool_choice is None
            or tool_choice == "auto"
            or getattr(tool_choice, "mode", None) == "auto"
        ):
            return None

        from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

        try:
            return get_model_structural_tag(
                model=self.structural_tag_model,
                tools=request.tools,
                tool_choice=request.tool_choice,
                reasoning=reasoning,
                token_suffix=self._parser_engine.token_suffix,
                strict_level=strict_level,
            )
        except Exception:
            logger.warning(
                "HYV4ToolParser failed to build a structural tag; falling back "
                "to the default guided-decoding path.",
                exc_info=True,
            )
            return None
