# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`sanitize_control_tokens` for renderers that build prompts without a Jinja
chat template, using the same markers as `apply_chat_template`."""

from typing import Any

from transformers.utils.chat_template_utils import (
    _get_control_token_patterns,
    encode_marked_chat,
    mark_control_tokens,
)

from .protocol import TokenizerLike


def _get_patterns(tokenizer: TokenizerLike) -> tuple[Any, Any]:
    # Without a template to scan, every added token counts as a control token
    control_tokens = tuple(
        token for token in tokenizer.get_added_vocab() if not token.isspace()
    )
    return _get_control_token_patterns(control_tokens)


def mark_chat_control_tokens(tokenizer: TokenizerLike, obj: Any) -> Any:
    """Mark the control tokens in the strings of `obj`, such as messages."""
    return mark_control_tokens(obj, _get_patterns(tokenizer)[0])


def encode_marked_prompt(tokenizer: TokenizerLike, prompt: str) -> list[int]:
    """Encode a prompt rendered from marked inputs, with marked control tokens
    as plain text."""
    return encode_marked_chat(tokenizer, prompt, _get_patterns(tokenizer)[1])
