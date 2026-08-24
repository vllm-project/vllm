# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Field transform rules for converting proto field names/values
to match vLLM's Pydantic models (OpenAI API-compatible).

These transforms bridge the gap between proto limitations and Python types:
  - content_parts → content  (proto can't have two fields with the same name)
  - parameters_json → parameters  (proto can't represent dict[str, Any])
  - prompt → prompt  (proto oneof creates nesting that Python unions don't)
  - messages → messages  (MessageToDict omits unset optional fields)
"""

import json
from typing import Any


def flatten_completion_prompt(prompt_dict: Any) -> Any:
    """Convert a CompletionPrompt oneof dict to a raw prompt value.

    CompletionPrompt is a proto message with a oneof field that represents
    the various prompt formats in CompletionRequest.prompt:
      - text (str) → str
      - texts (CompletionPromptTexts) → list[str]
      - token_ids (TokenIdSequence) → list[int]
      - token_id_batches (CompletionPromptTokenIdBatch) → list[list[int]]
    """
    if not isinstance(prompt_dict, dict):
        return prompt_dict
    if "text" in prompt_dict:
        return prompt_dict["text"]
    if "texts" in prompt_dict:
        inner = prompt_dict["texts"].get("texts", [])
        return list(inner)
    if "token_ids" in prompt_dict:
        inner = prompt_dict["token_ids"].get("token_ids", [])
        return [int(x) for x in inner]
    if "token_id_batches" in prompt_dict:
        batches = prompt_dict["token_id_batches"].get("batches", [])
        return [[int(x) for x in b.get("token_ids", [])] for b in batches]
    raise ValueError("Invalid CompletionPrompt payload: no supported oneof field set")


def _parse_tool_choice(value: Any) -> Any:
    """Parse tool_choice from proto string to str or dict.

    tool_choice can be a simple string ("none", "auto", "required")
    or a JSON-encoded object (e.g. {"type": "function", "function": {"name": "..."}}).
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return value


def _ensure_message_content(messages: list) -> list:
    """Ensure each message has a 'content' key (default None).

    MessageToDict omits unset optional fields, but Pydantic
    requires explicit None for optional fields like content.
    """
    for msg in messages:
        if isinstance(msg, dict):
            msg.setdefault("content", None)
    return messages


# Format: {proto_field_name: (python_field_name, transform_fn_or_None)}
FIELD_TRANSFORMS: dict[str, tuple[str, Any]] = {
    "parameters_json": ("parameters", json.loads),
    "content_parts": ("content", None),
    "prompt": ("prompt", flatten_completion_prompt),
    "messages": ("messages", _ensure_message_content),
    "tool_choice": ("tool_choice", _parse_tool_choice),
}
