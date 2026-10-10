# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from vllm.exceptions import VLLMValidationError

# Passed in by user
ChatTemplateContentFormatOption = Literal["auto", "string", "openai"]


@dataclass
class ChatTemplateConfig:
    chat_template: str | None = None
    chat_template_content_format: ChatTemplateContentFormatOption = "auto"
    trust_request_chat_template: bool = False
    trust_request_mm_kwargs: bool = False


def validate_chat_template(chat_template: Path | str | None):
    """Raises if the provided chat template appears invalid."""
    if chat_template is None:
        return

    elif isinstance(chat_template, Path) and not chat_template.exists():
        raise FileNotFoundError("the supplied chat template path doesn't exist")

    elif isinstance(chat_template, str):
        JINJA_CHARS = "{}\n"
        if (
            not any(c in chat_template for c in JINJA_CHARS)
            and not Path(chat_template).exists()
        ):
            # Try to find the template in the built-in templates directory
            from vllm.transformers_utils.chat_templates.registry import (
                CHAT_TEMPLATES_DIR,
            )

            builtin_template_path = CHAT_TEMPLATES_DIR / chat_template
            if not builtin_template_path.exists():
                raise VLLMValidationError(
                    f"The supplied chat template string ({chat_template}) "
                    f"appears path-like, but doesn't exist! "
                    f"Tried: {chat_template} and {builtin_template_path}",
                    parameter="chat_template",
                )

    else:
        raise VLLMValidationError(
            f"{type(chat_template)} is not a valid chat template type",
            parameter="chat_template",
        )


def _load_chat_template(
    chat_template: Path | str | None,
    *,
    is_literal: bool = False,
) -> str | None:
    if chat_template is None:
        return None

    if is_literal:
        if isinstance(chat_template, Path):
            raise TypeError(
                "chat_template is expected to be read directly from its value"
            )

        return chat_template

    try:
        with open(chat_template) as f:
            return f.read()
    except OSError as e:
        if isinstance(chat_template, Path):
            raise

        JINJA_CHARS = "{}\n"
        if not any(c in chat_template for c in JINJA_CHARS):
            # Try to load from the built-in templates directory
            from vllm.transformers_utils.chat_templates.registry import (
                CHAT_TEMPLATES_DIR,
            )

            builtin_template_path = CHAT_TEMPLATES_DIR / chat_template
            try:
                with open(builtin_template_path) as f:
                    return f.read()
            except OSError:
                msg = (
                    f"The supplied chat template ({chat_template}) "
                    f"looks like a file path, but it failed to be opened. "
                    f"Tried: {chat_template} and {builtin_template_path}. "
                    f"Reason: {e}"
                )
                raise VLLMValidationError(msg, parameter="chat_template") from e

        # If opening a file fails, set chat template to be args to
        # ensure we decode so our escape are interpreted correctly
        return _load_chat_template(chat_template, is_literal=True)


_cached_load_chat_template = lru_cache(_load_chat_template)


def load_chat_template(
    chat_template: Path | str | None,
    *,
    is_literal: bool = False,
) -> str | None:
    return _cached_load_chat_template(chat_template, is_literal=is_literal)
