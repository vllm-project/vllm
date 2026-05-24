# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN56 — vllm#41466 backport: qwen3coder XML parse fallback.

Backport of upstream PR #41466 (ToastyTheBot, OPEN). When
`_parse_xml_function_call` throws or returns None inside
`extract_tool_calls_streaming`, the original code leaves
`prev_tool_call_arr[i]["arguments"]` with the placeholder `"{}"`
from the header-sent step. Serving layer's remaining-args check
later double-emits `{"arguments":"{}"}`, breaking strict OpenAI
clients (Vercel AI SDK, OpenAI Node SDK).

Composes with our existing P64 (vllm#39598) — P64 changed the
post-`except` flow but did NOT modify the try block where PN56
inserts. Anchor stable on both pristine and post-P64 file states.

Sub-patches (2):
  A — insert `parse_succeeded = False/True` flags around the try
  B — append fallback block: when parse_succeeded=False, restore
      `prev_tool_call_arr[i]["arguments"]` from
      `streamed_args_for_tool[i] + "}"` (matches what the serving
      layer's remainder check will compute).

Affects ALL Genesis configs that use qwen3_coder tool parser:
- 27B Lorbus + MTP K=3 + tools (PROD)
- 27B FP8 short/long + tools
- 35B FP8 + DFlash/MTP + tools

Default OFF — defensive backport. Risk: low (extra branch per parse,
data-only fallback). Enable after live verify against tool-call sweep.

Author: Sandermage backport (ToastyTheBot, vllm#41466).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn56_qwen3coder_xml_fallback")

GENESIS_PN56_MARKER = "Genesis PN56 qwen3coder XML parse fallback (vllm#41466)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN56_QWEN3CODER_XML_FALLBACK", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# Sub-A: wrap try with parse_succeeded flag
ANCHOR_A_OLD = (
    "                if func_content_end != -1:\n"
    "                    func_content = tool_text[func_start:func_content_end]\n"
    "                    try:\n"
    "                        parsed_tool = self._parse_xml_function_call(\n"
    "                            func_content,\n"
    "                        )\n"
    "                        if parsed_tool and self.current_tool_index < len(\n"
    "                            self.prev_tool_call_arr\n"
    "                        ):\n"
    "                            self.prev_tool_call_arr[self.current_tool_index][\n"
    "                                \"arguments\"\n"
    "                            ] = parsed_tool.function.arguments\n"
    "                    except Exception:\n"
    "                        logger.debug(\n"
    "                            \"Failed to parse tool call during streaming: %s\",\n"
    "                            tool_text,\n"
    "                            exc_info=True,\n"
    "                        )"
)

ANCHOR_A_NEW = (
    "                if func_content_end != -1:\n"
    "                    func_content = tool_text[func_start:func_content_end]\n"
    "                    # [Genesis PN56 vllm#41466] Track parse success to know\n"
    "                    # if fallback below should fire (else \"{}\" placeholder leaks).\n"
    "                    _pn56_parse_succeeded = False\n"
    "                    try:\n"
    "                        parsed_tool = self._parse_xml_function_call(\n"
    "                            func_content,\n"
    "                        )\n"
    "                        if parsed_tool and self.current_tool_index < len(\n"
    "                            self.prev_tool_call_arr\n"
    "                        ):\n"
    "                            self.prev_tool_call_arr[self.current_tool_index][\n"
    "                                \"arguments\"\n"
    "                            ] = parsed_tool.function.arguments\n"
    "                            _pn56_parse_succeeded = True\n"
    "                    except Exception:\n"
    "                        logger.debug(\n"
    "                            \"Failed to parse tool call during streaming: %s\",\n"
    "                            tool_text,\n"
    "                            exc_info=True,\n"
    "                        )\n"
    "                    # [Genesis PN56 vllm#41466] When parse failed, prev_tool_call_arr\n"
    "                    # still has \"{}\" placeholder. Restore from incrementally\n"
    "                    # streamed args + closing brace so serving layer remainder\n"
    "                    # check produces correct output instead of double-emit \"{}\".\n"
    "                    if (\n"
    "                        not _pn56_parse_succeeded\n"
    "                        and self.current_tool_index < len(self.prev_tool_call_arr)\n"
    "                        and self.current_tool_index < len(self.streamed_args_for_tool)\n"
    "                    ):\n"
    "                        # [Audit A-14 fix 2026-05-05] Guard against double `}}`\n"
    "                        # if streamed_args already ends with closing brace\n"
    "                        # (P64 may have written it, or a prior partial close).\n"
    "                        _pn56_streamed = self.streamed_args_for_tool[\n"
    "                            self.current_tool_index\n"
    "                        ]\n"
    "                        _pn56_suffix = \"\" if _pn56_streamed.rstrip().endswith(\"}\") else \"}\"\n"
    "                        self.prev_tool_call_arr[self.current_tool_index][\n"
    "                            \"arguments\"\n"
    "                        ] = _pn56_streamed + _pn56_suffix"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("tool_parsers/qwen3coder_tool_parser.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN56 qwen3coder XML fallback (vllm#41466)",
        target_file=str(target),
        marker=GENESIS_PN56_MARKER,
        sub_patches=[TextPatch(
            name="pn56_xml_fallback",
            anchor=ANCHOR_A_OLD,
            replacement=ANCHOR_A_NEW,
            required=True,
        )],
        upstream_drift_markers=[
            "_pn56_parse_succeeded",
            "parse_succeeded = False",
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN56")
    log_decision("PN56", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "qwen3coder_tool_parser.py not found"
    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", "PN56 applied: XML parse failure no longer leaks {} placeholder"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return "skipped", f"{msg} — likely upstream merged or P64 reshaped block"
    return "failed", failure.reason if failure else "unknown failure"
