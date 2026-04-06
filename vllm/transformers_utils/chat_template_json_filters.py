# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extra Jinja filters for chat templates (e.g. Gemma 4 tool JSON expansion)."""

from __future__ import annotations

import json
from typing import Any


def _tool_response_json_normalize(value: Any) -> Any:
    """If *value* is a JSON object/array string, parse it for template formatting.

    Used by examples/tool_chat_template_gemma4.jinja so tool message ``content``
    like ``{"current_working_directory": "/alex"}`` expands to structured
    ``response:name{key:...}`` instead of a single ``value:`` string.
    """
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def patch_transformers_chat_template_jinja_json_filters() -> None:
    """Register ``tool_response_json_normalize`` on Jinja sandbox environments.

    HuggingFace compiles chat templates with ``ImmutableSandboxedEnvironment``;
    filters must exist before ``from_string``, so we extend ``__init__``.
    """
    import jinja2.sandbox

    cls = jinja2.sandbox.ImmutableSandboxedEnvironment
    if getattr(cls, "_vllm_tool_response_json_normalize_patched", False):
        return

    _orig_init = cls.__init__

    def _patched_init(self, *args: Any, **kwargs: Any) -> None:
        _orig_init(self, *args, **kwargs)
        self.filters["tool_response_json_normalize"] = _tool_response_json_normalize

    cls.__init__ = _patched_init  # type: ignore[assignment]
    cls._vllm_tool_response_json_normalize_patched = True  # type: ignore[attr-defined]
