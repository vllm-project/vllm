# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from openai.types.responses import ToolChoiceCustom, ToolChoiceFunction
from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    ConstStringFormat,
    OrFormat,
    SequenceFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TokenFormat,
    TriggeredTagsFormat,
)

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.registered_adapters import Gemma4ParserToolAdapter
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel


class Gemma4EngineToolParser(Gemma4ParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named = False
    structural_tag_model = "gemma_4"

    def get_structural_tag(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
        *,
        reasoning: bool = False,
        strict_level: ToolStrictLevel = ToolStrictLevel.AUTO,
    ) -> StructuralTag | None:
        tag = super().get_structural_tag(
            request, reasoning=reasoning, strict_level=strict_level
        )
        if (
            tag is None
            or request.tool_choice != "required"
            or not isinstance(tag.format, TriggeredTagsFormat)
        ):
            return tag

        # The required branch is also composed with a response schema: no free text.
        start, end = "<|tool_call>", "<tool_call|>"
        calls = [
            SequenceFormat(
                elements=[
                    ConstStringFormat(value=call.begin.removeprefix(start)),
                    call.content,
                ]
            )
            for call in tag.format.tags
        ]
        # Preflight has no tokenizer, so token nodes must already contain IDs.
        tag.format = TagsWithSeparatorFormat(
            tags=[
                TagFormat(
                    begin=TokenFormat(token=self.vocab[start]),
                    content=OrFormat(elements=calls),
                    end=TokenFormat(token=self.vocab[end]),
                )
            ],
            separator="",
            at_least_one=True,
            stop_after_first=tag.format.stop_after_first,
        )
        return tag

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Skip structured-output JSON for required/named tool choice.

        Gemma4 emits its native `<|tool_call>call:...` syntax, which the
        parser extracts directly. The base `ToolParser.adjust_request` would
        set `structured_outputs` for required/named and force JSON via guided
        decoding, conflicting with that native syntax (it leaks as content and
        crashes EngineCore under speculative decoding). Skip it so the model
        emits its native format (mirrors the GLM4 parser).
        """
        if request.tools:
            tc = request.tool_choice
            if tc == "required" or isinstance(
                tc,
                (
                    ChatCompletionNamedToolChoiceParam,
                    ToolChoiceFunction,
                    ToolChoiceCustom,
                ),
            ):
                request.skip_special_tokens = False
                return request
        return super().adjust_request(request)
