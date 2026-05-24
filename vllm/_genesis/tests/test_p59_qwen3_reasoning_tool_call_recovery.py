# SPDX-License-Identifier: Apache-2.0
"""TDD for Patch 59 — Qwen3 reasoning embedded tool_call recovery.

Backport of vllm-project/vllm#39055 (ZenoAFfectionate).

Validates:
  1. Anchors against a synthetic file mimicking the post-P12 layout
  2. Idempotency (second apply is no-op)
  3. Upstream-drift detection (skip when `_split_embedded_tool_calls` already present)
  4. Opt-in env-flag gating

Behavioural validation happens via blue/green container reproducer test —
see Genesis_Doc/spec_decode_investigation/v7_12_session/.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

from pathlib import Path

import pytest


SYNTHETIC_PARSER_FILE = (
    "# GENESIS_PATCHED -- Qwen3 tool_call reasoning fix v2 (PR #35687 full mirror)\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
    "\n"
    "from collections.abc import Sequence\n"
    "from typing import TYPE_CHECKING\n"
    "\n"
    "from vllm.entrypoints.openai.engine.protocol import DeltaMessage\n"
    "from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser\n"
    "\n"
    "if TYPE_CHECKING:\n"
    "    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest\n"
    "    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest\n"
    "    from vllm.tokenizers import TokenizerLike\n"
    "\n"
    "\n"
    "class Qwen3ReasoningParser(BaseThinkingReasoningParser):\n"
    "    \"\"\"Reasoning parser for the Qwen3/Qwen3.5 model family.\"\"\"\n"
    "\n"
    "    @property\n"
    "    def start_token(self) -> str:\n"
    "        \"\"\"The token that starts reasoning content.\"\"\"\n"
    "        return \"<think>\"\n"
    "\n"
    "    @property\n"
    "    def end_token(self) -> str:\n"
    "        \"\"\"The token that ends reasoning content.\"\"\"\n"
    "        return \"</think>\"\n"
    "\n"
    "    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:\n"
    "        return False\n"
    "\n"
    "    def extract_reasoning(self, model_output, request):\n"
    "        # [Genesis v5.12] PR #35687: 3-way branch with <tool_call>\n"
    "        if self.end_token in model_output:\n"
    "            reasoning, _, content = model_output.partition(self.end_token)\n"
    "            return reasoning, content or None\n"
    "\n"
    "        if not self.thinking_enabled:\n"
    "            return None, model_output\n"
    "\n"
    "        # Thinking enabled but no </think>: output was truncated.\n"
    "        # Everything generated so far is reasoning.\n"
    "        return model_output, None\n"
)


@pytest.fixture
def fake_parser_file(tmp_path):
    p = tmp_path / "qwen3_reasoning_parser.py"
    p.write_text(SYNTHETIC_PARSER_FILE)
    return str(p)


def _make_p59_patcher(target_file: str, marker_suffix: str):
    from vllm._genesis.wiring.text_patch import TextPatcher, TextPatch
    from vllm._genesis.wiring.structured_output.patch_59_qwen3_reasoning_tool_call_recovery import (
        IMPORT_OLD, IMPORT_NEW,
        REGEX_OLD, REGEX_NEW,
        METHOD_OLD, METHOD_NEW,
        RETURN_THINK_MONOLITH_OLD, RETURN_THINK_MONOLITH_NEW,
        RETURN_THINK_MODULAR_OLD, RETURN_THINK_MODULAR_NEW,
        RETURN_TRUNC_OLD, RETURN_TRUNC_NEW,
    )
    return TextPatcher(
        patch_name="P59 test",
        target_file=target_file,
        marker=f"P59_TEST_{marker_suffix}",
        sub_patches=[
            TextPatch(name="import", anchor=IMPORT_OLD, replacement=IMPORT_NEW, required=True),
            TextPatch(name="regex", anchor=REGEX_OLD, replacement=REGEX_NEW, required=True),
            TextPatch(name="method", anchor=METHOD_OLD, replacement=METHOD_NEW, required=True),
            TextPatch(name="think_monolith", anchor=RETURN_THINK_MONOLITH_OLD, replacement=RETURN_THINK_MONOLITH_NEW, required=False),
            TextPatch(name="think_modular", anchor=RETURN_THINK_MODULAR_OLD, replacement=RETURN_THINK_MODULAR_NEW, required=False),
            TextPatch(name="trunc", anchor=RETURN_TRUNC_OLD, replacement=RETURN_TRUNC_NEW, required=False),
        ],
        upstream_drift_markers=["_split_embedded_tool_calls"],
    )


class TestP59AllAnchorsHit:
    def test_all_5_anchors_present_in_synthetic_file(self, fake_parser_file):
        # P59 was split in the 2026-04-25 refactor: the single
        # RETURN_THINK_OLD anchor became MONOLITH/MODULAR variants to
        # cover both pre-/post-#36138 layouts (P62). Verify the variant
        # that the synthetic file is shaped for matches.
        from vllm._genesis.wiring.structured_output.patch_59_qwen3_reasoning_tool_call_recovery import (
            IMPORT_OLD, REGEX_OLD, METHOD_OLD,
            RETURN_THINK_MONOLITH_OLD, RETURN_THINK_MODULAR_OLD,
            RETURN_TRUNC_OLD,
        )
        content = Path(fake_parser_file).read_text()
        # Required anchors must always be present.
        for name, anchor in [
            ("IMPORT", IMPORT_OLD),
            ("REGEX", REGEX_OLD),
            ("METHOD", METHOD_OLD),
            ("RETURN_TRUNC", RETURN_TRUNC_OLD),
        ]:
            assert content.count(anchor) == 1, (
                f"{name}_OLD anchor must appear exactly once "
                f"(got {content.count(anchor)})"
            )
        # Exactly ONE of the THINK variants must match (pre-/post-#36138).
        think_hits = (
            content.count(RETURN_THINK_MONOLITH_OLD)
            + content.count(RETURN_THINK_MODULAR_OLD)
        )
        assert think_hits == 1, (
            f"Exactly one of RETURN_THINK_MONOLITH_OLD or "
            f"RETURN_THINK_MODULAR_OLD must match (got {think_hits})"
        )


class TestP59Application:
    def test_apply_inserts_helper_method_and_regex(self, fake_parser_file):
        from vllm._genesis.wiring.text_patch import TextPatchResult
        patcher = _make_p59_patcher(fake_parser_file, "APPLY")
        result, failure = patcher.apply()
        assert result == TextPatchResult.APPLIED, failure
        modified = Path(fake_parser_file).read_text()
        assert "import re  # [Genesis P59 vllm#39055]" in modified
        assert "_EMBEDDED_TOOL_CALL_RE = re.compile" in modified
        assert "_split_embedded_tool_calls" in modified
        assert "self._split_embedded_tool_calls(reasoning, content or None)" in modified
        assert "self._split_embedded_tool_calls(model_output, None)" in modified

    def test_modified_file_parses_as_python(self, fake_parser_file):
        import ast
        from vllm._genesis.wiring.text_patch import TextPatchResult
        patcher = _make_p59_patcher(fake_parser_file, "PARSE")
        result, _ = patcher.apply()
        assert result == TextPatchResult.APPLIED
        ast.parse(Path(fake_parser_file).read_text())  # raises if invalid


class TestP59Idempotency:
    def test_second_apply_is_idempotent(self, fake_parser_file):
        from vllm._genesis.wiring.text_patch import TextPatchResult
        patcher = _make_p59_patcher(fake_parser_file, "IDEMP")
        r1, _ = patcher.apply()
        r2, _ = patcher.apply()
        assert r1 == TextPatchResult.APPLIED
        assert r2 == TextPatchResult.IDEMPOTENT


class TestP59UpstreamDriftDetection:
    def test_skip_when_upstream_marker_present(self, tmp_path):
        from vllm._genesis.wiring.text_patch import (
            TextPatcher, TextPatch, TextPatchResult,
        )
        post_fix = tmp_path / "post_fix_parser.py"
        post_fix.write_text(
            "# already merged upstream\n"
            "def _split_embedded_tool_calls(): pass\n"
        )
        patcher = TextPatcher(
            patch_name="P59 drift",
            target_file=str(post_fix),
            marker="P59_DRIFT_TEST",
            sub_patches=[
                TextPatch(name="x", anchor="placeholder",
                          replacement="x", required=True),
            ],
            upstream_drift_markers=["_split_embedded_tool_calls"],
        )
        result, failure = patcher.apply()
        assert result == TextPatchResult.SKIPPED
        assert failure.reason == "upstream_merged"


class TestP59OptIn:
    def test_apply_skips_without_env_flag(self, monkeypatch):
        monkeypatch.delenv("GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY", raising=False)
        from vllm._genesis.wiring.structured_output.patch_59_qwen3_reasoning_tool_call_recovery import (
            apply,
        )
        status, reason = apply()
        assert status == "skipped"
        assert "opt-in" in reason

    def test_env_flag_truthy_returns_true(self, monkeypatch):
        monkeypatch.setenv("GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY", "1")
        from vllm._genesis.wiring.structured_output.patch_59_qwen3_reasoning_tool_call_recovery import (
            _is_enabled,
        )
        assert _is_enabled() is True
