# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vLLM source code checksum utility."""

from pathlib import Path

import pytest

from vllm.compilation.source_checksum import (
    calculate_vllm_source_checksum,
    calculate_vllm_source_factors,
    should_exclude_dir,
)


def test_should_exclude_dir():
    """Test directory exclusion logic."""
    # Should exclude directories starting with dot
    assert should_exclude_dir(".git")
    assert should_exclude_dir(".github")
    assert should_exclude_dir(".pytest_cache")

    # Should exclude cache directories
    assert should_exclude_dir("__pycache__")
    assert should_exclude_dir(".mypy_cache")
    assert should_exclude_dir(".ruff_cache")

    # Should not exclude normal directories
    assert not should_exclude_dir("vllm")
    assert not should_exclude_dir("csrc")
    assert not should_exclude_dir("tests")
    assert not should_exclude_dir("benchmarks")


def test_calculate_vllm_source_checksum_consistency():
    """Test that checksum calculation is consistent across multiple runs."""
    factors1 = calculate_vllm_source_factors()
    checksum1 = calculate_vllm_source_checksum(factors1)

    factors2 = calculate_vllm_source_factors()
    checksum2 = calculate_vllm_source_checksum(factors2)

    # Checksums should be identical for the same codebase
    assert checksum1 == checksum2
    # Checksum should be a SHA256 hex string (64 characters)
    assert len(checksum1) == 64

    # Factors should also be identical
    assert factors1 == factors2


def test_calculate_vllm_source_factors_structure():
    """Test that factors are properly structured."""
    factors = calculate_vllm_source_factors()

    # Should be a dict
    assert isinstance(factors, dict)

    # Should have some files
    assert len(factors) > 0

    # All keys should be relative paths (strings)
    for path in factors:
        assert isinstance(path, str)
        assert not path.startswith("/")  # Should be relative

    # All values should be SHA256 checksums (64 chars)
    for checksum in factors.values():
        assert isinstance(checksum, str)
        assert len(checksum) == 64


def test_includes_main_source_directories():
    """Test that main source directories are included."""
    factors = calculate_vllm_source_factors()

    # Should include files from vllm/
    vllm_files = [p for p in factors if p.startswith("vllm/")]
    assert len(vllm_files) > 0

    # Should include files from csrc/ if it exists
    csrc_files = [p for p in factors if p.startswith("csrc/")]
    # csrc directory should exist and have files
    assert len(csrc_files) > 0


def test_checksum_changes_with_factors():
    """Test that different factors produce different checksums."""
    factors1 = {"file1.py": "a" * 64}
    factors2 = {"file1.py": "b" * 64}

    checksum1 = calculate_vllm_source_checksum(factors1)
    checksum2 = calculate_vllm_source_checksum(factors2)

    assert checksum1 != checksum2


def test_excludes_dot_directories():
    """Test that directories starting with dot are excluded."""
    factors = calculate_vllm_source_factors()

    # No file should be in a dot directory
    for path in factors:
        parts = path.split("/")
        for part in parts:
            if part:  # Skip empty parts
                assert not part.startswith("."), f"Found file in dot directory: {path}"


def test_factors_change_when_file_added():
    """Test that factors change when a new source file is added."""

    # Get vLLM root directory
    current = Path(__file__).resolve()
    root_path = None
    for parent in current.parents:
        if (parent / "setup.py").is_file() and (parent / "vllm").is_dir():
            root_path = parent
            break

    assert root_path is not None

    # Calculate initial factors
    factors_before = calculate_vllm_source_factors()
    checksum_before = calculate_vllm_source_checksum(factors_before)

    # Create a temporary test file in vllm/
    test_file = root_path / "vllm" / "_test_checksum_temporary.py"
    assert not test_file.exists(), "Temporary test file already exists"

    try:
        # Add a new Python file
        test_file.write_text("# Temporary test file for checksum testing\n")

        # Calculate factors after adding file
        factors_after = calculate_vllm_source_factors()
        checksum_after = calculate_vllm_source_checksum(factors_after)

        # Factors should be different
        assert factors_before != factors_after
        # New file should be in factors
        assert "vllm/_test_checksum_temporary.py" in factors_after
        # Old file should not have been in factors
        assert "vllm/_test_checksum_temporary.py" not in factors_before
        # Checksums should be different
        assert checksum_before != checksum_after

        # Modify the file
        test_file.write_text("# Modified temporary test file\n")

        # Calculate factors after modification
        factors_modified = calculate_vllm_source_factors()
        checksum_modified = calculate_vllm_source_checksum(factors_modified)

        # File checksum should change
        assert (
            factors_after["vllm/_test_checksum_temporary.py"]
            != factors_modified["vllm/_test_checksum_temporary.py"]
        )
        # Overall checksum should change
        assert checksum_after != checksum_modified

    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()

    # Verify we're back to original state
    factors_final = calculate_vllm_source_factors()
    checksum_final = calculate_vllm_source_checksum(factors_final)

    assert factors_final == factors_before
    assert checksum_final == checksum_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
