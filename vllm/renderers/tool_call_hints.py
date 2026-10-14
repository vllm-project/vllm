# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render example tool calls the way the chat template writes them.

The prompt carries each tool's schema, but the model emits the call syntax of
its chat template, which is a different token sequence. Applying the template
to the conversation with and without an example assistant tool call appended
and taking the difference gives that syntax in the model's own tokens. The
``ngram_hint`` speculative method drafts from the result.
"""

import json
from typing import Any, cast

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam, ConversationMessage
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)


def _placeholder(schema: Any) -> Any:
    """Return a value of the type the parameter schema declares."""
    kind = schema.get("type", "string") if isinstance(schema, dict) else "string"
    if kind in ("integer", "number"):
        return 0
    if kind == "boolean":
        return True
    if kind == "array":
        return []
    if kind == "object":
        return {}
    return "X"


def _example_arguments(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function", tool)
    parameters = function.get("parameters") or {}
    properties = parameters.get("properties") or {}
    return {name: _placeholder(schema) for name, schema in properties.items()}


def render_tool_call_hints(
    tokenizer: TokenizerLike,
    conversation: list[ConversationMessage],
    tools: list[dict[str, Any]] | None,
    chat_template: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[list[int]]:
    """Return the token ids of one example call per tool, as the chat template
    renders it after the conversation.

    The argument values are placeholders; only the syntax around them is meant
    to be drafted. A tool whose call the template cannot render is skipped.
    """
    if not tools:
        return []

    kwargs: dict[str, Any] = dict(chat_template_kwargs or {})
    kwargs.pop("add_generation_prompt", None)
    if chat_template is not None:
        kwargs["chat_template"] = chat_template

    def render(messages: list[ConversationMessage]) -> str:
        return cast(
            str,
            tokenizer.apply_chat_template(
                cast(list[ChatCompletionMessageParam], messages),
                tools=tools,
                add_generation_prompt=False,
                tokenize=False,
                **kwargs,
            ),
        )

    try:
        prefix = render(conversation)
    except Exception as e:
        logger.debug("Cannot render the conversation for tool call hints: %s", e)
        return []

    hints: list[list[int]] = []
    for tool in tools:
        function = tool.get("function", tool)
        name = function.get("name")
        if not name:
            continue

        example_call = cast(
            ConversationMessage,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "0",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(_example_arguments(tool)),
                        },
                    }
                ],
            },
        )
        try:
            rendered = render([*conversation, example_call])
        except Exception as e:
            logger.debug("Cannot render a tool call hint for %r: %s", name, e)
            continue

        if not rendered.startswith(prefix):
            logger.debug("Tool call hint for %r does not extend the prompt", name)
            continue
        suffix = rendered[len(prefix) :]
        if not suffix:
            continue

        # The hint continues the prompt, so no BOS. Special tokens in the suffix
        # are kept as single ids, which is the default for template output.
        token_ids = tokenizer.encode(suffix, add_special_tokens=False)
        if token_ids:
            hints.append(list(token_ids))

    return hints
