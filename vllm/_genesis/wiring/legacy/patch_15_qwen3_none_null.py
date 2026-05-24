# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 15 — Qwen3 chat-template `None` vs `null` tool-call fix.

Problem
-------
Qwen3.5+ chat templates use Jinja's `| string` filter for scalar tool-call
arguments, which produces Python's `repr()` form `None` instead of the JSON
literal `null`. The qwen3coder tool parser only recognises lowercase `null`,
so a `None` slips through as the literal string `"None"` and breaks any
tool with a nullable parameter.

Reference: vLLM PR [#38996](https://github.com/vllm-project/vllm/pull/38996)
            issue [#38885](https://github.com/vllm-project/vllm/issues/38885).

Fix
---
Accept both `null` and `none` (case-insensitive) in `_convert_param_value`:

    # before:
    if param_value.lower() == "null":
        return None
    # after:
    if param_value.lower() in ("null", "none"):
        return None

Platform compatibility: vendor-agnostic — pure Python parser logic.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p15_qwen3_none_null")

GENESIS_P15_MARKER = "Genesis P15 Qwen3 None/null tool arg v7.0"

UPSTREAM_DRIFT_MARKERS = [
    # If PR #38996 merges, the file will contain the multi-value tuple.
    '("null", "none")',
    "'null', 'none'",
]


_OLD = (
    "        # Handle null value for any type\n"
    '        if param_value.lower() == "null":\n'
    "            return None"
)

_NEW = (
    "        # [Genesis P15] Handle null/none value for any type (PR #38996).\n"
    "        # Qwen3.5+ chat template emits Python repr 'None' (Jinja `| string`)\n"
    "        # instead of JSON 'null'. Accept both case-insensitively.\n"
    '        if param_value.lower() in ("null", "none"):\n'
    "            return None"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("tool_parsers/qwen3coder_tool_parser.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P15 Qwen3 None/null tool arg",
        target_file=target,
        marker=GENESIS_P15_MARKER,
        sub_patches=[
            TextPatch(
                name="p15_none_null",
                anchor=_OLD,
                replacement=_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def apply() -> tuple[str, str]:
    """Apply P15 wiring. Never raises."""
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "qwen3coder_tool_parser.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", "None/none mapping added to tool param parser"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    return "failed", failure.reason if failure else "unknown failure"
