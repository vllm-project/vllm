# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from .registry import (
    get_chat_template_fallback_path,
    get_chat_template_override_path,
)

__all__ = [
    "get_chat_template_fallback_path",
    "get_chat_template_override_path",
]
