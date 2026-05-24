# SPDX-License-Identifier: Apache-2.0
"""TDD for the text-patch wiring primitive.

Each test operates on a temp-file so we don't touch any real vLLM source.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import pytest


@pytest.fixture
def fake_source(tmp_path):
    """Create a tmp Python-like source file for patching."""
    path = tmp_path / "fake_module.py"
    path.write_text(
        "# fake module for tests\n"
        "\n"
        "def foo():\n"
        "    return 1\n"
        "\n"
        "def bar():\n"
        "    raise NotImplementedError('no hybrid')\n"
        "\n"
        "def baz():\n"
        "    return 3\n"
    )
    return str(path)


class TestTextPatcherBasics:
    def test_applies_single_sub_patch(self, fake_source):
        from vllm._genesis.wiring.text_patch import (
            TextPatcher, TextPatch, TextPatchResult,
        )

        p = TextPatcher(
            patch_name="test-basic",
            target_file=fake_source,
            marker="test_marker_BASIC",
            sub_patches=[
                TextPatch(
                    name="edit_bar",
                    anchor="    raise NotImplementedError('no hybrid')\n",
                    replacement="    return 42\n",
                    required=True,
                ),
            ],
        )

        result, failure = p.apply()
        assert result == TextPatchResult.APPLIED
        assert failure is None

        content = open(fake_source).read()
        assert "return 42" in content
        assert "no hybrid" not in content
        assert "test_marker_BASIC" in content  # marker present

    def test_idempotent_on_second_call(self, fake_source):
        from vllm._genesis.wiring.text_patch import (
            TextPatcher, TextPatch, TextPatchResult,
        )

        p = TextPatcher(
            patch_name="test-idemp",
            target_file=fake_source,
            marker="IDEMPOTENT_XYZ",
            sub_patches=[
                TextPatch(name="e", anchor="return 3\n",
                          replacement="return 333\n", required=True),
            ],
        )

        r1, _ = p.apply()
        r2, _ = p.apply()
        assert r1 == TextPatchResult.APPLIED
        assert r2 == TextPatchResult.IDEMPOTENT

        # Content should only have one 'return 333' — not re-applied twice
        content = open(fake_source).read()
        assert content.count("return 333") == 1

    def test_missing_file_is_skip(self, tmp_path):
        from vllm._genesis.wiring.text_patch import (
            TextPatcher, TextPatch, TextPatchResult,
        )

        p = TextPatcher(
            patch_name="test-nofile",
            target_file=str(tmp_path / "nonexistent.py"),
            marker="M",
            sub_patches=[TextPatch("x", "a", "b", required=True)],
        )

        result, failure = p.apply()
        assert result == TextPatchResult.SKIPPED
        assert failure.reason == "target_file_missing"

    def test_missing_required_anchor_is_skip(self, fake_source):
        """Required anchor missing → safe skip, NOT crash, NOT partial write."""
        from vllm._genesis.wiring.text_patch import (
            TextPatcher, TextPatch, TextPatchResult,
        )

        before_content = open(fake_source).read()

        p = TextPatcher(
            patch_name="test-miss",
            target_file=fake_source,
            marker="M_MISSING",
            sub_patches=[
                TextPatch(
                    name="missing_anchor",
                    anchor="this text is definitely not in the file XYZ123",
                    replacement="replacement",
                    required=True,
                ),
            ],
        )

        result, failure = p.apply()
        assert result == TextPatchResult.SKIPPED
        assert failure.reason == "required_anchor_missing"

        # File untouched — no partial write, no marker
        after_content = open(fake_source).read()
        assert after_content == before_content
        assert "M_MISSING" not in after_content

    def test_optional_anchor_missing_allows_siblings(self, fake_source):
        """Non-required anchor missing → skip just that sub-patch, apply others."""
        from vllm._genesis.wiring.text_patch import (
            TextPatcher, TextPatch, TextPatchResult,
        )

        p = TextPatcher(
            patch_name="test-opt",
            target_file=fake_source,
            marker="M_OPTMIX",
            sub_patches=[
                TextPatch(
                    name="will_miss",
                    anchor="this is not in the file",
                    replacement="doesnt matter",
                    required=False,
                ),
                TextPatch(
                    name="will_apply",
                    anchor="return 1\n",
                    replacement="return 111\n",
                    required=True,
                ),
            ],
        )

        result, _ = p.apply()
        assert result == TextPatchResult.APPLIED

        content = open(fake_source).read()
        assert "return 111" in content
        assert "M_OPTMIX" in content

    def test_ambiguous_anchor_is_skip(self, tmp_path):
        """Anchor appearing multiple times → skip, don't replace blindly."""
        from vllm._genesis.wiring.text_patch import (
            TextPatcher, TextPatch, TextPatchResult,
        )

        path = tmp_path / "ambig.py"
        path.write_text("x = 1\nx = 1\nx = 2\n")

        p = TextPatcher(
            patch_name="test-ambig",
            target_file=str(path),
            marker="M_AMBIG",
            sub_patches=[
                TextPatch("a", "x = 1\n", "x = 999\n", required=True),
            ],
        )

        result, failure = p.apply()
        assert result == TextPatchResult.SKIPPED
        assert failure.reason == "ambiguous_anchor"

        # No partial write
        assert open(path).read() == "x = 1\nx = 1\nx = 2\n"

    def test_upstream_drift_marker_skips(self, fake_source):
        """If upstream merged a fix marker, skip our patch cleanly."""
        from vllm._genesis.wiring.text_patch import (
            TextPatcher, TextPatch, TextPatchResult,
        )

        # Add upstream marker to file
        path = fake_source
        content = open(path).read()
        open(path, "w").write("# upstream_fix_ABC merged\n" + content)

        p = TextPatcher(
            patch_name="test-drift",
            target_file=path,
            marker="OUR_M",
            sub_patches=[
                TextPatch("e", "return 1\n", "return 9\n", required=True),
            ],
            upstream_drift_markers=["upstream_fix_ABC merged"],
        )

        result, failure = p.apply()
        assert result == TextPatchResult.SKIPPED
        assert failure.reason == "upstream_merged"

        # No replacement done
        assert "return 1" in open(path).read()
        assert "return 9" not in open(path).read()


class TestTextPatcherRealWorldScenario:
    """Integration-like: multi-step patch resembling P4."""

    def test_multi_step_like_p4(self, tmp_path):
        from vllm._genesis.wiring.text_patch import (
            TextPatcher, TextPatch, TextPatchResult,
        )

        # Simulate arg_utils.py-like structure
        path = tmp_path / "args.py"
        path.write_text(
            "import stuff\n"
            "\n"
            "\n"
            "@dataclass\n"
            "class EngineArgs:\n"
            "    def create_engine_config(self):\n"
            "        if hybrid:\n"
            "            raise NotImplementedError(\"not supported\")\n"
            "        return config\n"
        )

        # Two sub-patches: inject helper BEFORE @dataclass, then rewrite raise
        helper_src = (
            "\n\ndef _genesis_helper():\n"
            "    return []\n"
        )

        p = TextPatcher(
            patch_name="P4-sim",
            target_file=str(path),
            marker="P4_SIM_MARKER",
            sub_patches=[
                TextPatch(
                    name="inject_helper",
                    anchor="\n\n@dataclass\nclass EngineArgs:",
                    replacement=helper_src + "\n\n@dataclass\nclass EngineArgs:",
                    required=True,
                ),
                TextPatch(
                    name="rewrite_raise",
                    anchor="        if hybrid:\n            raise NotImplementedError(\"not supported\")\n",
                    replacement="        if hybrid:\n            pass  # allowed\n",
                    required=True,
                ),
            ],
        )

        result, _ = p.apply()
        assert result == TextPatchResult.APPLIED

        content = open(path).read()
        assert "_genesis_helper" in content
        assert "pass  # allowed" in content
        assert "not supported" not in content
        assert "P4_SIM_MARKER" in content

        # Re-apply is idempotent
        r2, _ = p.apply()
        assert r2 == TextPatchResult.IDEMPOTENT
