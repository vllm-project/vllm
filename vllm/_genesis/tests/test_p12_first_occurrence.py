# SPDX-License-Identifier: Apache-2.0
"""TDD test for C1 fix — P12 should emit FIRST occurrence of <tool_call>,
not LAST. Originally P61 was supposed to fix this but its anchor doesn't
match P12-emitted code (see project_genesis_quality_audit_20260428).

The fix: update P12's _NEW_METHODS_BLOCK to use input_ids.index(...) rather
than input_ids[::-1].index(...) — preserves multi-tool agentic flows that
emit multiple <tool_call> blocks where the FIRST should be the content
boundary, not the LAST.

This test inspects the patch source string directly (no vllm import needed)
so it runs in CI without GPUs.
"""
from __future__ import annotations

import re


from vllm._genesis.wiring.legacy.patch_12_tool_call_reasoning import _NEW_METHODS_BLOCK


def test_p12_emits_first_occurrence_not_last():
    """C1 fix: P12 must emit FIRST tool_call index, not LAST.

    Multi-tool agentic flows (Qwen3 tool-calling) emit multiple <tool_call>
    blocks. The boundary between reasoning and content is the FIRST
    <tool_call>, not the last. Original P12 returned LAST (intended for
    SINGLE-tool case) which silently drops intermediate tool calls.

    P61 was supposed to fix this via anchor-replacement but its anchor
    `tool_call_index = ...` doesn't match P12's emitted `idx = ...` —
    silent skip. C1 fixes P12 directly, retires P61.
    """
    # Anti-pattern (LAST): input_ids[::-1].index(...) — should NOT appear
    last_pattern = "input_ids[::-1].index"
    assert last_pattern not in _NEW_METHODS_BLOCK, (
        f"P12 emits LAST-occurrence pattern {last_pattern!r}. "
        f"This drops intermediate tool calls in multi-tool flows. "
        f"Replace with input_ids.index(...) for FIRST occurrence."
    )

    # Correct pattern (FIRST): input_ids.index(self._tool_call_token_id)
    # without the [::-1] reverse
    first_pattern = re.compile(
        r"input_ids\.index\(\s*self\._tool_call_token_id\s*\)"
    )
    assert first_pattern.search(_NEW_METHODS_BLOCK), (
        "P12 should emit FIRST-occurrence pattern "
        "`input_ids.index(self._tool_call_token_id)` for the multi-tool "
        "boundary detection. Pattern not found in _NEW_METHODS_BLOCK."
    )


def test_p12_preserves_existing_first_occurrence_callers():
    """Defense: P12's other code paths (is_reasoning_end, extract_reasoning)
    must remain functional. Smoke check that we still find the expected
    helper symbols in the emit block."""
    expected_methods = (
        "is_reasoning_end",
        "extract_content_ids",
        "extract_reasoning",
    )
    for method_name in expected_methods:
        assert method_name in _NEW_METHODS_BLOCK, (
            f"P12 emit lost {method_name} method (refactor must preserve it)"
        )
