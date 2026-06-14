# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for use_existing_torch.py to verify the three stripping modes:
  - default (no flag): removes ALL lines containing the word 'torch'
  - --prefix:          removes lines whose content starts with a torch lib prefix
  - --torch-only:      removes only the core torch version pin, preserving
                       torchvision and torchaudio version pins

Regression test for: GH200 CI build failure where run-gh200-test.sh called
use_existing_torch.py without any flag, causing torchvision and comment lines
to be incorrectly stripped from requirements/cuda.txt, breaking the Docker
build for arm64 (GH200) targets.
"""

import os
import sys
import textwrap

import pytest

# Add repo root to path so we can import use_existing_torch directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from use_existing_torch import main

SAMPLE_CUDA_TXT = textwrap.dedent("""\
    # Common dependencies
    -r common.txt

    # Dependencies for NVIDIA GPUs
    torch==2.10.0
    torchaudio==2.10.0
    # These must be updated alongside torch
    torchvision==0.25.0 # Required for phi3v processor

    # FlashInfer
    flashinfer-python==0.6.6
""")

SAMPLE_PYPROJECT = textwrap.dedent("""\
    [project]
    dependencies = [
        "torch == 2.10.0",
        "torchvision == 0.25.0",
    ]
    no-build-isolation-package = ["torch"]
""")


@pytest.fixture
def tmp_requirements(tmp_path):
    """Create a temporary directory with sample requirements and pyproject files."""
    req_dir = tmp_path / "requirements"
    req_dir.mkdir()
    (req_dir / "cuda.txt").write_text(SAMPLE_CUDA_TXT)
    (tmp_path / "pyproject.toml").write_text(SAMPLE_PYPROJECT)
    return tmp_path


def test_torch_only_mode_strips_only_torch_pin(tmp_requirements, monkeypatch):
    """--torch-only mode should remove only the core torch version pin.

    torchvision and torchaudio version pins must be preserved so they are
    still installed from requirements files (critical for GH200 arm64 builds
    where torchvision is needed for the phi3v processor).
    """
    monkeypatch.chdir(tmp_requirements)
    main(["--torch-only"])

    cuda_content = (tmp_requirements / "requirements" / "cuda.txt").read_text()
    # Core torch version pin must be removed
    assert "torch==2.10.0" not in cuda_content
    # torchvision and torchaudio pins must be PRESERVED
    assert "torchaudio==2.10.0" in cuda_content
    assert "torchvision==0.25.0" in cuda_content
    # Comment lines must be preserved
    assert "# These must be updated alongside torch" in cuda_content
    # Unrelated deps preserved
    assert "flashinfer-python==0.6.6" in cuda_content

    pyproject_content = (tmp_requirements / "pyproject.toml").read_text()
    assert '"torch == 2.10.0"' not in pyproject_content
    # torchvision pin preserved
    assert '"torchvision == 0.25.0"' in pyproject_content
    # no-build-isolation-package line preserved
    assert 'no-build-isolation-package = ["torch"]' in pyproject_content


def test_prefix_mode_strips_all_torch_lib_pins(tmp_requirements, monkeypatch):
    """--prefix mode should remove torch, torchvision, and torchaudio version pins.

    Comment lines and lines that only contain the word 'torch' as part of
    another token (e.g. no-build-isolation-package) must be preserved.
    """
    monkeypatch.chdir(tmp_requirements)
    main(["--prefix"])

    cuda_content = (tmp_requirements / "requirements" / "cuda.txt").read_text()
    # All three lib version pins removed
    assert "torch==2.10.0" not in cuda_content
    assert "torchaudio==2.10.0" not in cuda_content
    assert "torchvision==0.25.0" not in cuda_content
    # Comment line preserved (does not start with a torch lib prefix)
    assert "# These must be updated alongside torch" in cuda_content
    # Unrelated deps preserved
    assert "flashinfer-python==0.6.6" in cuda_content

    pyproject_content = (tmp_requirements / "pyproject.toml").read_text()
    assert '"torch == 2.10.0"' not in pyproject_content
    assert '"torchvision == 0.25.0"' not in pyproject_content
    # no-build-isolation-package line preserved
    assert 'no-build-isolation-package = ["torch"]' in pyproject_content


def test_no_flag_mode_strips_all_torch_lines(tmp_requirements, monkeypatch):
    """Without any flag, ALL lines containing the word 'torch' are removed."""
    monkeypatch.chdir(tmp_requirements)
    main([])

    cuda_content = (tmp_requirements / "requirements" / "cuda.txt").read_text()
    assert "torch==2.10.0" not in cuda_content
    assert "torchaudio==2.10.0" not in cuda_content
    assert "torchvision==0.25.0" not in cuda_content
    # Comment line also removed (contains 'torch')
    assert "# These must be updated alongside torch" not in cuda_content
    # Unrelated lines preserved
    assert "flashinfer-python==0.6.6" in cuda_content


def test_gh200_ci_uses_torch_only_flag():
    """Regression test: GH200 CI script must call use_existing_torch.py --torch-only.

    Using no flag or --prefix incorrectly strips torchvision from
    requirements/cuda.txt, causing the phi3v processor to be unavailable
    in the resulting Docker image.
    """
    script_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        ".buildkite",
        "scripts",
        "hardware_ci",
        "run-gh200-test.sh",
    )
    with open(script_path) as f:
        content = f.read()

    # Every non-comment invocation of use_existing_torch.py must use --torch-only
    lines_with_script = [
        line
        for line in content.splitlines()
        if "use_existing_torch.py" in line and not line.strip().startswith("#")
    ]
    assert lines_with_script, "run-gh200-test.sh must invoke use_existing_torch.py"
    for line in lines_with_script:
        assert "--torch-only" in line, (
            f"run-gh200-test.sh calls use_existing_torch.py without "
            f"--torch-only: {line!r}\n"
            "This causes torchvision to be stripped, breaking the GH200 Docker build."
        )
