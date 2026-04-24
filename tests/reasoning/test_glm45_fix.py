"""
Unit tests for the glm45 embedded tool-call promotion fix.
Tests _split_embedded_tool_calls in isolation — no model needed.

Run inside the container:
    python3 /tmp/glm45_fix/test_glm45_fix.py

Or from workspace:
    python3 /workspace/workspace/glm45_parser_fix/test_glm45_fix.py
"""

import sys
import re

# ── inline the regex patterns and static method so we can test without vLLM ──

_EMBEDDED_TOOL_CALL_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>.*$",
    re.DOTALL,
)

_BODY_WITHOUT_OPEN_RE = re.compile(
    r"^(?:[\w.-]+\n?)?"
    r"(?:<arg_key>.*?</arg_key>"
    r"<arg_value>.*?</arg_value>)*"
    r"<arg_key>.*?</arg_key>"
    r"<arg_value>.*?"
    r"(?:</arg_value>)?"
    r"</tool_call>"
    r".*$",
    re.DOTALL,
)

_CLOSING_FRAGMENT_RE = re.compile(
    r"^[^<]*</arg_value>\s*</tool_call>\s*$",
    re.DOTALL,
)


def _split_embedded_tool_calls(
    reasoning: str | None,
    content: str | None,
) -> tuple[str | None, str | None]:
    if not reasoning:
        return reasoning, content

    promoted: list[str] = []
    phase_counts = [0, 0, 0]

    def _collect_phase1(m: re.Match) -> str:
        promoted.append(m.group(0))
        phase_counts[0] += 1
        return ""

    cleaned = _EMBEDDED_TOOL_CALL_RE.sub(_collect_phase1, reasoning).strip()

    # Phase 2: runs on Phase-1 remainder even if Phase 1 found something
    if cleaned and _BODY_WITHOUT_OPEN_RE.match(cleaned):
        reconstructed = "<tool_call>" + cleaned
        promoted.append(reconstructed)
        phase_counts[1] += 1
        cleaned = ""

    # Phase 3: closing fragment
    if cleaned and _CLOSING_FRAGMENT_RE.match(cleaned):
        if content and "<tool_call>" in content:
            n_open = content.count("<tool_call>")
            n_close = content.count("</tool_call>")
            if n_open > n_close:
                content = content + cleaned
                phase_counts[2] += 1
                cleaned = ""
                if not promoted:
                    return None, content
        elif not content:
            # 3b: no content — promote fragment so tool parser can attempt recovery
            promoted.append(cleaned)
            phase_counts[2] += 1
            cleaned = ""

    if not promoted:
        return reasoning, content

    promoted_text = "\n".join(promoted)
    merged_content = (
        promoted_text if not content else promoted_text + "\n" + content
    )

    return cleaned or None, merged_content


# ── helpers ───────────────────────────────────────────────────────────────────

def run_test(name: str, fn):
    try:
        fn()
        print(f"PASS {name}")
        return True
    except AssertionError as e:
        print(f"FAIL {name}: {e}")
        return False


# ── Phase 1 tests: full embedded <tool_call>...</tool_call> ───────────────────

def test_no_tool_call_unchanged():
    """Pure reasoning prose must pass through unmodified."""
    r = "This is just reasoning about the task."
    reasoning, content = _split_embedded_tool_calls(r, None)
    assert reasoning == r, f"Expected unchanged reasoning, got: {reasoning!r}"
    assert content is None


def test_tool_call_promoted_from_reasoning():
    """A <tool_call> block inside reasoning must be moved to content."""
    tool_xml = "<tool_call>bash<arg_key>command</arg_key><arg_value>ls /</arg_value></tool_call>"
    reasoning = f"I should list the files.\n{tool_xml}"
    reasoning_out, content_out = _split_embedded_tool_calls(reasoning, None)

    assert tool_xml not in (reasoning_out or ""), \
        "tool_call must not remain in reasoning"
    assert content_out is not None and tool_xml in content_out, \
        f"tool_call must appear in content, got: {content_out!r}"
    assert "I should list the files." in (reasoning_out or ""), \
        "prose reasoning must be preserved"


def test_promoted_prepended_to_existing_content():
    """Promoted tool call must be prepended to any existing content."""
    tool_xml = "<tool_call>bash<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>"
    reasoning = f"Let me run this.\n{tool_xml}"
    existing_content = "Some trailing content after </think>."

    _, content_out = _split_embedded_tool_calls(reasoning, existing_content)
    assert content_out.startswith(tool_xml), \
        f"Promoted block must be first in content, got: {content_out!r}"
    assert existing_content in content_out, \
        "Existing content must be preserved"


def test_multiple_tool_calls_promoted():
    """Multiple embedded tool calls must all be promoted."""
    t1 = "<tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"
    t2 = "<tool_call>bash<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>"
    reasoning = f"Step 1.\n{t1}\nStep 2.\n{t2}"

    reasoning_out, content_out = _split_embedded_tool_calls(reasoning, None)
    assert t1 in content_out and t2 in content_out, \
        f"Both tool calls must be in content, got: {content_out!r}"
    assert t1 not in (reasoning_out or "") and t2 not in (reasoning_out or ""), \
        "Neither tool call must remain in reasoning"


def test_truncated_tool_call_promoted():
    """A truncated (incomplete) <tool_call> at end-of-string must also be promoted."""
    truncated = "<tool_call>bash<arg_key>command</arg_key>"
    reasoning = f"I will run:\n{truncated}"

    reasoning_out, content_out = _split_embedded_tool_calls(reasoning, None)
    assert content_out is not None and truncated in content_out, \
        f"Truncated tool call must be promoted, got: {content_out!r}"


def test_none_reasoning_passthrough():
    """None reasoning must return unchanged."""
    reasoning_out, content_out = _split_embedded_tool_calls(None, "some content")
    assert reasoning_out is None
    assert content_out == "some content"


def test_empty_reasoning_after_promotion_returns_none():
    """If reasoning contains only tool calls, cleaned reasoning must be None."""
    tool_xml = "<tool_call>bash<arg_key>command</arg_key><arg_value>echo hi</arg_value></tool_call>"
    reasoning_out, _ = _split_embedded_tool_calls(tool_xml, None)
    assert reasoning_out is None, \
        f"All-tool-call reasoning must become None, got: {reasoning_out!r}"


# ── Phase 2 tests: body-without-open-tag (special token stripped by vLLM) ────

def test_phase2_body_without_open_tag_simple():
    """
    The <tool_call> special token (id 154829) is stripped by vLLM.
    reasoning_content contains the body but not the opening <tool_call> tag.
    The function must prepend <tool_call> and promote to content.

    Observed failure (from AIVLLM-229 trace):
        reasoning_content='bash<arg_key>command</arg_key>
                           <arg_value>grep -n "def distance" /testbed/sympy/...</arg_value>
                           </tool_call>'
    """
    body = (
        "bash"
        "<arg_key>command</arg_key>"
        "<arg_value>grep -n \"def distance\" /testbed/sympy/geometry/point.py</arg_value>"
        "</tool_call>"
    )
    reasoning_out, content_out = _split_embedded_tool_calls(body, None)

    assert content_out is not None, "content must not be None"
    assert content_out.startswith("<tool_call>"), \
        f"reconstructed block must start with <tool_call>, got: {content_out!r}"
    assert "bash" in content_out, "function name must be preserved"
    assert "grep" in content_out, "argument value must be preserved"
    assert reasoning_out is None, \
        f"reasoning must be empty after promotion, got: {reasoning_out!r}"


def test_phase2_body_without_open_multikey():
    """
    Multi-key body-without-open (str_replace_editor pattern from trace):
        reasoning_content='str_replace_editor
                           <arg_key>command</arg_key><arg_value>view</arg_value>
                           <arg_key>path</arg_key><arg_value>/testbed/...</arg_value>
                           <arg_key>view_range</arg_key><arg_value>[1, 100]</arg_value>
                           </tool_call>'
    """
    body = (
        "str_replace_editor"
        "<arg_key>command</arg_key><arg_value>view</arg_value>"
        "<arg_key>path</arg_key><arg_value>/testbed/sympy/geometry/point.py</arg_value>"
        "<arg_key>view_range</arg_key><arg_value>[1, 100]</arg_value>"
        "</tool_call>"
    )
    reasoning_out, content_out = _split_embedded_tool_calls(body, None)

    assert content_out is not None and content_out.startswith("<tool_call>"), \
        f"multi-key body must be reconstructed, got: {content_out!r}"
    assert "str_replace_editor" in content_out
    assert "view_range" in content_out
    assert reasoning_out is None


def test_phase2_body_with_prose_preserved():
    """
    If reasoning has both prose and a body-without-open, only the body
    portion (at the end, containing </tool_call>) should match.
    Because the whole reasoning string doesn't match _BODY_WITHOUT_OPEN_RE,
    Phase 2 should NOT fire — the prose must be left intact.

    This is a false-positive guard: we don't promote unless the entire
    reasoning (after Phase 1 stripping) matches the body pattern.
    """
    reasoning = "Let me think about this carefully.\nbash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"
    # The whole string doesn't match _BODY_WITHOUT_OPEN_RE from ^ anchor
    # (prose before the body breaks the pattern), so Phase 2 should not fire.
    # The behavior here: Phase 1 also doesn't match (no <tool_call> open tag).
    # Result: reasoning unchanged, content None.
    # NOTE: This tests the false-positive guard — in practice the model emits
    # body-without-open as the *entire* reasoning_content (prose is separate).
    reasoning_out, content_out = _split_embedded_tool_calls(reasoning, None)
    # Prose mixed with body: neither phase 1 nor phase 2 matches the full string,
    # so reasoning passes through unchanged.
    assert reasoning_out == reasoning, \
        f"Mixed prose+body must not be promoted, got reasoning_out={reasoning_out!r}"
    assert content_out is None


def test_phase2_truncated_body_without_open():
    """
    A truncated body-without-open (partial arg_value, no close):
        'bash<arg_key>command</arg_key><arg_value>grep ...'
    Phase 2 should NOT fire because there is no </tool_call> closing tag.
    The truncated text passes through unchanged.
    """
    truncated_body = "bash<arg_key>command</arg_key><arg_value>grep -n def"
    reasoning_out, content_out = _split_embedded_tool_calls(truncated_body, None)
    # No </tool_call> — Phase 2 anchor not satisfied
    assert reasoning_out == truncated_body, \
        f"Truncated body without </tool_call> must not be promoted, got: {reasoning_out!r}"
    assert content_out is None


# ── Phase 3 tests: closing fragment only ─────────────────────────────────────

def test_phase3_closing_fragment_appended_to_content():
    """
    The call head is already in content (unclosed), tail leaked into reasoning.

    Observed failure (from AIVLLM-229 trace):
        reasoning_content='</arg_value></tool_call>'
        content='<tool_call>bash<arg_key>command</arg_key><arg_value>ls -la'

    The fragment must be appended to content; reasoning must become None.
    """
    fragment = "</arg_value></tool_call>"
    content_head = "<tool_call>bash<arg_key>command</arg_key><arg_value>ls -la"

    reasoning_out, content_out = _split_embedded_tool_calls(fragment, content_head)

    assert reasoning_out is None, \
        f"reasoning must be cleared after Phase 3, got: {reasoning_out!r}"
    assert content_out == content_head + fragment, \
        f"fragment must be appended to content, got: {content_out!r}"
    assert content_out.endswith("</tool_call>"), \
        "merged content must end with </tool_call>"


def test_phase3_closing_fragment_with_value_suffix():
    """
    Fragment contains the end of arg_value + closing tags:
        '0.0, 0.0, 0.0</arg_value></tool_call>'

    Observed failure (first failure from AIVLLM-229 trace):
        reasoning_content='0.0, 0.0, 0.0</arg_value></tool_call>'
    """
    fragment = "0.0, 0.0, 0.0</arg_value></tool_call>"
    content_head = "<tool_call>distance<arg_key>p1</arg_key><arg_value>"

    reasoning_out, content_out = _split_embedded_tool_calls(fragment, content_head)

    assert reasoning_out is None, \
        f"reasoning must be cleared after Phase 3, got: {reasoning_out!r}"
    assert content_out == content_head + fragment, \
        f"fragment must be appended to content, got: {content_out!r}"


def test_phase3_no_fire_without_unclosed_content():
    """
    Phase 3 must NOT fire if content has no unmatched <tool_call> open.
    The closing fragment is left unchanged.
    """
    fragment = "</arg_value></tool_call>"
    # Content has matching open+close — no unclosed call
    balanced_content = "<tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"

    reasoning_out, content_out = _split_embedded_tool_calls(fragment, balanced_content)
    # Phase 3 guard: n_open == n_close, so don't fire
    assert reasoning_out == fragment, \
        f"Phase 3 must not fire when content has no unclosed call, got: {reasoning_out!r}"
    assert content_out == balanced_content


def test_phase3_no_fire_without_tool_call_in_content():
    """Phase 3 must NOT fire if content has no <tool_call> at all."""
    fragment = "</arg_value></tool_call>"
    plain_content = "This is just plain text."

    reasoning_out, content_out = _split_embedded_tool_calls(fragment, plain_content)
    assert reasoning_out == fragment
    assert content_out == plain_content


def test_phase3_no_fire_with_none_content():
    """Phase 3 must NOT fire if content is None."""
    fragment = "</arg_value></tool_call>"
    reasoning_out, content_out = _split_embedded_tool_calls(fragment, None)
    assert reasoning_out == fragment
    assert content_out is None


# ── Phase 1 + Phase 2 interaction: second call body after Phase 1 strip ──────

def test_phase1_then_phase2_on_remainder():
    """
    Key production scenario (AIVLLM-229):
    Model emits two concatenated <tool_call> blocks inside <think>, but the
    second call's <tool_call> open token (id 154829) is stripped by vLLM.

    reasoning:
      <tool_call>bash<arg_key>command</arg_key><arg_value>grep</arg_value></tool_call>
      bash<arg_key>command</arg_key><arg_value>pytest\n271</arg_value></tool_call>

    Phase 1 promotes the first (complete) block.
    Phase 2 must then reconstruct the second from the remainder.
    Both must appear in content.
    """
    t1 = "<tool_call>bash<arg_key>command</arg_key><arg_value>grep -n def</arg_value></tool_call>"
    body2 = "bash<arg_key>command</arg_key><arg_value>pytest -k distance\n271</arg_value></tool_call>"
    reasoning = t1 + body2

    reasoning_out, content_out = _split_embedded_tool_calls(reasoning, None)

    assert content_out is not None, "content must not be None"
    assert t1 in content_out, f"Phase 1 call must be in content, got: {content_out!r}"
    assert "<tool_call>" + body2 in content_out, \
        f"Phase 2 reconstructed call must be in content, got: {content_out!r}"
    assert reasoning_out is None, \
        f"reasoning must be empty after both phases, got: {reasoning_out!r}"
    assert content_out.count("<tool_call>") == 2, \
        f"must have exactly 2 <tool_call> blocks, got: {content_out!r}"


# ── Phase 3b: closing fragment with content=None ──────────────────────────────

def test_phase3b_fragment_promoted_when_content_none():
    """
    Production scenario: model stops at stop_token=154829 (<tool_call>) inside
    <think>, so there is no </think> and content=None.  Only the closing tail
    of the previous call is in reasoning_content.

    Observed:
        reasoning_content='\n   271</arg_value></tool_call>'   content=None
        reasoning_content='0, 0, 0)</arg_value></tool_call>'   content=None

    The fragment must be promoted to content even without an unclosed call
    in content (since content doesn't exist), so the tool parser can see it.
    """
    fragment = "\n   271</arg_value></tool_call>"
    reasoning_out, content_out = _split_embedded_tool_calls(fragment, None)

    assert content_out is not None, \
        "fragment must be promoted to content even when content was None"
    assert "</tool_call>" in content_out, \
        f"promoted content must contain closing tag, got: {content_out!r}"
    assert reasoning_out is None, \
        f"reasoning must be cleared, got: {reasoning_out!r}"


def test_phase3b_value_suffix_fragment_content_none():
    """
    Observed: reasoning_content='0, 0, 0)</arg_value></tool_call>'  content=None
    """
    fragment = "0, 0, 0)</arg_value></tool_call>"
    reasoning_out, content_out = _split_embedded_tool_calls(fragment, None)

    assert content_out is not None and "</tool_call>" in content_out, \
        f"fragment must be promoted, got: {content_out!r}"
    assert reasoning_out is None


# ── regression: phases must not interfere with each other ────────────────────

def test_phase1_does_not_double_wrap():
    """
    If Phase 1 matches (full <tool_call> present), Phase 2 must not
    double-wrap it after Phase 1 already promoted it.
    """
    full_block = "<tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"
    reasoning_out, content_out = _split_embedded_tool_calls(full_block, None)

    # Phase 1 should handle it, resulting in one <tool_call> in content
    assert content_out is not None and content_out.count("<tool_call>") == 1, \
        f"Phase 1 must produce exactly one block, got: {content_out!r}"
    # No double-wrapping
    assert "<tool_call><tool_call>" not in (content_out or ""), \
        "Phase 2 must not double-wrap a Phase 1 result"


# ── entry point ───────────────────────────────────────────────────────────────

TESTS = [
    # Phase 1
    ("test_no_tool_call_unchanged",                test_no_tool_call_unchanged),
    ("test_tool_call_promoted_from_reasoning",     test_tool_call_promoted_from_reasoning),
    ("test_promoted_prepended_to_existing_content",test_promoted_prepended_to_existing_content),
    ("test_multiple_tool_calls_promoted",          test_multiple_tool_calls_promoted),
    ("test_truncated_tool_call_promoted",          test_truncated_tool_call_promoted),
    ("test_none_reasoning_passthrough",            test_none_reasoning_passthrough),
    ("test_empty_reasoning_after_promotion",       test_empty_reasoning_after_promotion_returns_none),
    # Phase 2
    ("test_phase2_body_without_open_simple",       test_phase2_body_without_open_tag_simple),
    ("test_phase2_body_without_open_multikey",     test_phase2_body_without_open_multikey),
    ("test_phase2_body_with_prose_preserved",      test_phase2_body_with_prose_preserved),
    ("test_phase2_truncated_body_no_close",        test_phase2_truncated_body_without_open),
    # Phase 1 + 2 interaction
    ("test_phase1_then_phase2_on_remainder",       test_phase1_then_phase2_on_remainder),
    # Phase 3a (unclosed call in content)
    ("test_phase3_closing_fragment_appended",      test_phase3_closing_fragment_appended_to_content),
    ("test_phase3_fragment_with_value_suffix",     test_phase3_closing_fragment_with_value_suffix),
    ("test_phase3_no_fire_balanced_content",       test_phase3_no_fire_without_unclosed_content),
    ("test_phase3_no_fire_no_tool_call_content",   test_phase3_no_fire_without_tool_call_in_content),
    # Phase 3b (content=None — observed in production with stop_reason=154829)
    ("test_phase3b_fragment_content_none",         test_phase3b_fragment_promoted_when_content_none),
    ("test_phase3b_value_suffix_content_none",     test_phase3b_value_suffix_fragment_content_none),
    # Regression
    ("test_phase1_no_double_wrap",                 test_phase1_does_not_double_wrap),
    # Phase 3 no-fire with None content is now changed behavior:
    # Phase 3b PROMOTES when content=None, so remove the old no-fire test for None
]


if __name__ == "__main__":
    failed = 0
    for name, fn in TESTS:
        if not run_test(name, fn):
            failed += 1
    total = len(TESTS)
    print(f"\n{total - failed}/{total} tests passed." +
          (f"  {failed} FAILED." if failed else "  All green."))
    sys.exit(failed)
