# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that error messages in Anthropic and speech-to-text entrypoints
are sanitized to prevent memory address leakage.

Verifies the fix for the incomplete CVE-2026-22778 remediation where
PIL repr addresses leaked via the Anthropic API router and the
speech-to-text WebSocket paths.
"""

import pytest

from vllm.entrypoints.serve.exception_handling.utils import sanitize_message


def test_sanitize_message():
    assert (
        sanitize_message("<_io.BytesIO object at 0x7a95e299e750>")
        == "<_io.BytesIO object>"
    )


class TestSanitizeMessageFilePaths:
    """sanitize_message should also strip file paths and traceback
    frames, not just memory addresses - see #31683."""

    def test_strips_traceback_style_frame(self):
        msg = (
            "1 validation error:\n"
            "  {'type': 'list_type', 'loc': ('body', 'messages')}\n"
            '\n  File "/usr/local/lib/python3.12/dist-packages/vllm/'
            'entrypoints/serve/utils/api_utils.py", line 40, '
            "in create_chat_completion\n"
            "    POST /v1/chat/completions"
        )
        result = sanitize_message(msg)
        assert "/usr/local/" not in result
        assert "api_utils.py" not in result
        assert "list_type" in result

    def test_strips_arbitrary_absolute_path(self):
        result = sanitize_message("Error in /home/user/project/vllm/server.py")
        assert "/home/user" not in result

    def test_strips_single_parent_container_path(self):
        """Regression: /app/server.py and /workspace/server.py (common in
        container deployments) were missed by the original {2,} quantifier."""
        assert "/app/" not in sanitize_message("Error in /app/server.py")
        assert "/workspace/" not in sanitize_message("Error in /workspace/server.py")

    def test_preserves_api_endpoint_paths(self):
        msg = "POST /v1/chat/completions failed"
        assert "/v1/chat/completions" in sanitize_message(msg)

    def test_preserves_short_field_references(self):
        msg = "Invalid value for field 'body.messages'"
        assert sanitize_message(msg) == msg

    def test_strips_both_address_and_path(self):
        msg = (
            "<Request at 0x7f123> failed at "
            "/usr/local/lib/python3.12/dist-packages/vllm/server.py"
        )
        result = sanitize_message(msg)
        assert "0x" not in result
        assert "/usr/local/" not in result


class TestSanitizeMessageCoversLeakPatterns:
    """Ensure sanitize_message strips addresses from realistic exceptions."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (
                "cannot identify image file <_io.BytesIO object at 0x7a95e299e750>",
                "cannot identify image file <_io.BytesIO object>",
            ),
            (
                "cannot identify image file <_io.BytesIO object at 0x7f3c1a2b4d90>",
                "cannot identify image file <_io.BytesIO object>",
            ),
            (
                "<PIL.PngImagePlugin.PngImageFile image mode=RGB "
                "size=8x8 at 0x7f3c1a2b4d90>",
                "<PIL.PngImagePlugin.PngImageFile image mode=RGB size=8x8>",
            ),
            (
                "Error processing <_io.BytesIO object at 0xdeadbeef>: invalid header",
                "Error processing <_io.BytesIO object>: invalid header",
            ),
        ],
        ids=[
            "bytesio-standard",
            "bytesio-different-addr",
            "pil-image-repr",
            "mid-string-repr",
        ],
    )
    def test_address_stripped(self, raw: str, expected: str):
        assert sanitize_message(raw) == expected

    def test_safe_message_unchanged(self):
        msg = "Invalid request: missing 'messages' field"
        assert sanitize_message(msg) == msg

    def test_multiple_addresses_stripped(self):
        raw = "<obj at 0xaaa> and <obj at 0xbbb>"
        result = sanitize_message(raw)
        assert "0x" not in result


class TestAffectedModulesUseSanitize:
    """Verify that affected modules call sanitize_message (source-level)."""

    @pytest.mark.parametrize(
        "module",
        [
            "vllm.entrypoints.anthropic.serving",
            "vllm.entrypoints.speech_to_text.realtime.connection",
        ],
    )
    def test_module_calls_sanitize_message(self, module: str):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.find_spec(module)
        assert spec is not None and spec.origin is not None, (
            f"Cannot locate module {module}"
        )
        source = Path(spec.origin).read_text()
        assert "sanitize_message" in source, f"{module} does not call sanitize_message"
        assert "import" in source and "sanitize_message" in source
