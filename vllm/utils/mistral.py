# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Provides lazy import of the vllm.tokenizers.mistral module."""

from __future__ import annotations

from random import choices
from string import ascii_letters, digits
from typing import TYPE_CHECKING, TypeGuard

from vllm.tokenizers import TokenizerLike
from vllm.utils.import_utils import LazyLoader

if TYPE_CHECKING:
    # if type checking, eagerly import the module
    import vllm.tokenizers.mistral as mt
else:
    mt = LazyLoader("mt", globals(), "vllm.tokenizers.mistral")

_MISTRAL_TOOL_CALL_ID_ALPHABET = ascii_letters + digits


def is_mistral_tokenizer(obj: TokenizerLike | None) -> TypeGuard[mt.MistralTokenizer]:
    """Return true if the tokenizer is a MistralTokenizer instance."""
    cls = type(obj)
    # Check for special class attribute, this avoids importing the class to
    # do an isinstance() check.  If the attribute is True, do an isinstance
    # check to be sure we have the correct type.
    return bool(
        getattr(cls, "IS_MISTRAL_TOKENIZER", False)
        and isinstance(obj, mt.MistralTokenizer)
    )


def generate_mistral_tool_call_id() -> str:
    """Generate a Mistral-compatible tool call ID."""
    # Mistral Tool Call Ids must be alphanumeric with a length of 9.
    # https://github.com/mistralai/mistral-common/blob/21ee9f6cee3441e9bb1e6ed2d10173f90bd9b94b/src/mistral_common/protocol/instruct/validator.py#L299
    # Keep this helper lightweight so generic streaming code does not need to
    # import the Mistral parser module at import time.
    return "".join(choices(_MISTRAL_TOOL_CALL_ID_ALPHABET, k=9))


def is_valid_mistral_tool_call_id(value: str) -> bool:
    """Validate a Mistral-compatible tool call ID."""
    return value.isalnum() and len(value) == 9
