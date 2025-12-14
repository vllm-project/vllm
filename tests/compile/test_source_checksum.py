# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import pytest

from vllm.compilation.source_checksum import (
    calculate_vllm_source_checksum,
    calculate_vllm_source_factors,
    should_exclude_dir,
)


def test_should_exclude_dir():
    assert should_exclude_dir(".git")
    assert should_exclude_dir(".github")
    assert should_exclude_dir(".pytest_cache")

    assert should_exclude_dir("__pycache__")
    assert should_exclude_dir(".mypy_cache")
    assert should_exclude_dir(".ruff_cache")

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

    assert checksum1 == checksum2
    # Checksum should be a SHA256 hex string (64 characters)
    assert len(checksum1) == 64

    assert factors1 == factors2


def test_calculate_vllm_source_factors_structure():
    factors = calculate_vllm_source_factors()

    assert isinstance(factors, dict)
    assert len(factors) > 0

    for path in factors:
        assert isinstance(path, str)
        assert not path.startswith("/")

    for checksum in factors.values():
        assert isinstance(checksum, str)
        assert len(checksum) == 64


def test_includes_main_source_directories():
    factors = calculate_vllm_source_factors()

    vllm_files = [p for p in factors if p.startswith("vllm/")]
    assert len(vllm_files) > 0

    csrc_files = [p for p in factors if p.startswith("csrc/")]
    assert len(csrc_files) > 0


def test_checksum_changes_with_factors():
    factors1 = {"file1.py": "a" * 64}
    factors2 = {"file1.py": "b" * 64}

    checksum1 = calculate_vllm_source_checksum(factors1)
    checksum2 = calculate_vllm_source_checksum(factors2)

    assert checksum1 != checksum2


def test_excludes_dot_directories():
    factors = calculate_vllm_source_factors()

    # No file should be in a dot directory
    for path in factors:
        parts = path.split("/")
        for part in parts:
            if part:  # Skip empty parts
                assert not part.startswith("."), f"Found file in dot directory: {path}"


def test_factors_change_when_file_added():
    current = Path(__file__).resolve()
    root_path = None
    for parent in current.parents:
        if (parent / "setup.py").is_file() and (parent / "vllm").is_dir():
            root_path = parent
            break

    assert root_path is not None

    factors_before = calculate_vllm_source_factors()
    checksum_before = calculate_vllm_source_checksum(factors_before)

    test_file = root_path / "vllm" / "_test_checksum_temporary.py"
    assert not test_file.exists(), "Temporary test file already exists"

    try:
        test_file.write_text("# Temporary test file for checksum testing\n")

        factors_after = calculate_vllm_source_factors()
        checksum_after = calculate_vllm_source_checksum(factors_after)

        assert factors_before != factors_after
        assert "vllm/_test_checksum_temporary.py" in factors_after
        assert "vllm/_test_checksum_temporary.py" not in factors_before
        assert checksum_before != checksum_after

        test_file.write_text("# Modified temporary test file\n")

        factors_modified = calculate_vllm_source_factors()
        checksum_modified = calculate_vllm_source_checksum(factors_modified)

        assert (
            factors_after["vllm/_test_checksum_temporary.py"]
            != factors_modified["vllm/_test_checksum_temporary.py"]
        )
        assert checksum_after != checksum_modified

    finally:
        if test_file.exists():
            test_file.unlink()

    factors_final = calculate_vllm_source_factors()
    checksum_final = calculate_vllm_source_checksum(factors_final)

    assert factors_final == factors_before
    assert checksum_final == checksum_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
