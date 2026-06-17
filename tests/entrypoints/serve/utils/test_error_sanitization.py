# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that error messages in Anthropic and speech-to-text entrypoints
are sanitized to prevent memory address leakage.

Verifies the fix for the incomplete CVE-2026-22778 remediation where
PIL repr addresses leaked via the Anthropic API router and the
speech-to-text WebSocket paths.
"""

import pytest

from vllm.entrypoints.serve.utils.api_utils import sanitize_message


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
            "vllm.entrypoints.anthropic.api_router",
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
