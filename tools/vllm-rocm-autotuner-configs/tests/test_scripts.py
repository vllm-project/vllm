# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test utility tools."""

import subprocess
import sys
from pathlib import Path

import pytest


class TestGenerateSignature:
    """Test generate_signature.py script."""

    def test_script_exists(self):
        script = Path(
            __file__).parent.parent / "tools" / "generate_signature.py"
        assert script.exists()

    def test_generate_signature(self, mock_model_dir):
        """Test signature generation."""
        script = Path(
            __file__).parent.parent / "tools" / "generate_signature.py"

        result = subprocess.run(
            [sys.executable, str(script),
             str(mock_model_dir)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Signature:" in result.stdout
        assert "GptOssForCausalLM_36L_2880H_64A" in result.stdout

    def test_missing_model_path(self):
        """Test handling of missing model."""
        script = Path(
            __file__).parent.parent / "tools" / "generate_signature.py"

        result = subprocess.run(
            [sys.executable, str(script), "/nonexistent/path"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Error" in result.stdout or "Error" in result.stderr


class TestAddSignatures:
    """Test add_signatures.py script."""

    def test_script_exists(self):
        script = Path(__file__).parent.parent / "tools" / "add_signatures.py"
        assert script.exists()

    @pytest.mark.slow
    def test_dry_run(self):
        """Test dry run mode."""
        script = Path(__file__).parent.parent / "tools" / "add_signatures.py"

        result = subprocess.run(
            [sys.executable, str(script), "--dry-run"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Dry run" in result.stdout or "dry run" in result.stdout.lower()
