# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Provides lazy import of the vllm.tokenizers.mistral module."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard

from vllm.tokenizers import TokenizerLike
from vllm.utils.import_utils import LazyLoader

if TYPE_CHECKING:
    # if type checking, eagerly import the module
    import vllm.tokenizers.mistral as mt
    import vllm.tool_parsers.mistral_tool_parser as mtp
else:
    mt = LazyLoader("mt", globals(), "vllm.tokenizers.mistral")
    mtp = LazyLoader("mtp", globals(), "vllm.tool_parsers.mistral_tool_parser")


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


def is_mistral_tool_parser(cls: type | None) -> bool:
    """Return true if *cls* is (a subclass of) MistralToolParser.

    Uses a class attribute check so that importing
    ``vllm.tool_parsers.mistral_tool_parser`` — and transitively
    ``mistral_common`` — is not required.
    """
    return bool(
        getattr(cls, "IS_MISTRAL_TOOL_PARSER", False)
        and issubclass(cls, mtp.MistralToolParser)  # type: ignore[arg-type]
    )
