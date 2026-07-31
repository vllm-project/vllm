# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.engine.protocol import (
        DeltaMessage,
        ExtractedToolCallInformation,
    )
    from vllm.tokenizers import TokenizerLike


class RustUnifiedToolParser(ToolParser):
    """Tool capability stub for Rust unified parsers."""

    rust_parser_name: ClassVar[str]

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

    def extract_tool_calls(
        self, model_output, request, **kwargs
    ) -> ExtractedToolCallInformation:
        raise NotImplementedError(
            "RustUnifiedToolParser is a capability stub. "
            "Use RustUnifiedParser for output parsing."
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request,
    ) -> DeltaMessage | None:
        raise NotImplementedError(
            "RustUnifiedToolParser is a capability stub. "
            "Use RustUnifiedParser for output parsing."
        )


def make_rust_unified_tool_parser(
    parser_name: str,
) -> type[RustUnifiedToolParser]:
    class_name = f"RustUnifiedToolParser_{parser_name}"
    parser_cls = type(
        class_name,
        (RustUnifiedToolParser,),
        {
            "__module__": __name__,
            "rust_parser_name": parser_name,
        },
    )
    globals()[class_name] = parser_cls
    return parser_cls
