# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.mcp.tool_server import ToolServer
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike


class RustUnifiedReasoningParser(ReasoningParser):
    """Reasoning capability stub for Rust unified parsers."""

    rust_parser_name: ClassVar[str]

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._rust_parser: Any | None = None

    def _get_parser(self) -> Any:
        if self._rust_parser is None:
            from vllm.parser.rust_unified_parser import (
                _tokenizer_metadata,
                rust_unified_parser_module,
            )

            self._rust_parser = rust_unified_parser_module().UnifiedParser(
                self.rust_parser_name,
                [],
                _tokenizer_metadata(self.model_tokenizer),
            )
        return self._rust_parser

    @property
    def reasoning_start_str(self) -> str | None:
        return self._get_parser().reasoning_start_str()

    @property
    def reasoning_end_str(self) -> str | None:
        return self._get_parser().reasoning_end_str()

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self._get_parser().is_reasoning_end(list(input_ids))

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        from vllm.parser.rust_unified_parser import _extract_content_ids

        return _extract_content_ids(
            self.model_tokenizer,
            self.reasoning_end_str,
            input_ids,
        )

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        return self._get_parser().count_reasoning_tokens(list(token_ids))

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        raise NotImplementedError(
            "RustUnifiedReasoningParser exposes reasoning capabilities. "
            "Use RustUnifiedParser for output parsing."
        )

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        raise NotImplementedError(
            "RustUnifiedReasoningParser is a capability stub. "
            "Use RustUnifiedParser for output parsing."
        )

    def prepare_structured_tag(
        self,
        original_tag: str | None,
        tool_server: ToolServer | None,
    ) -> str | None:
        return original_tag


def make_rust_unified_reasoning_parser(
    parser_name: str,
) -> type[RustUnifiedReasoningParser]:
    class_name = f"RustUnifiedReasoningParser_{parser_name}"
    parser_cls = type(
        class_name,
        (RustUnifiedReasoningParser,),
        {
            "__module__": __name__,
            "rust_parser_name": parser_name,
        },
    )
    globals()[class_name] = parser_cls
    return parser_cls
