# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression tests for nightly Transformers compatibility in model tests.

The Transformers Nightly CI job installs the latest transformers and then runs
test_initialization.py and test_transformers.py. Models with a
max_transformers_version cap (e.g. QWenLMHeadModel with
max_transformers_version="4.53") were previously skipped when running with a
nightly/dev transformers build because check_transformers_version() was called
with the default check_max_version=True.

The fix is to call check_transformers_version() with check_max_version=False in
both test_initialization.py and test_transformers.py so that these models are
exercised (not silently skipped) when running with nightly transformers.
"""
import pytest

import tests.models.registry as reg
from tests.models.registry import HF_EXAMPLE_MODELS


def _find_arch_with_max_version() -> str | None:
    """Return the first model arch that has a max_transformers_version set."""
    for arch in HF_EXAMPLE_MODELS.get_supported_archs():
        info = HF_EXAMPLE_MODELS.get_hf_info(arch)
        if info.max_transformers_version is not None:
            return arch
    return None


def test_check_transformers_version_respects_check_max_version_false():
    """check_transformers_version(check_max_version=False) must not skip a model
    even when the installed transformers version exceeds max_transformers_version.
    """
    arch = _find_arch_with_max_version()
    if arch is None:
        pytest.skip("No model with max_transformers_version found in registry")

    info = HF_EXAMPLE_MODELS.get_hf_info(arch)
    original_version = reg.TRANSFORMERS_VERSION
    try:
        # Simulate a nightly transformers version that exceeds the cap
        reg.TRANSFORMERS_VERSION = "99.0.0"
        # With check_max_version=False, this must return None (no skip)
        result = info.check_transformers_version(
            on_fail="return",
            check_max_version=False,
            check_version_reason="vllm",
        )
        assert result is None, (
            f"check_transformers_version(check_max_version=False) returned "
            f"{result!r} for {arch!r} with transformers==99.0.0. "
            "Expected None — the model should NOT be skipped."
        )
    finally:
        reg.TRANSFORMERS_VERSION = original_version


def test_check_transformers_version_skips_with_check_max_version_true():
    """Baseline: check_transformers_version(check_max_version=True) must skip a
    model when the installed transformers version exceeds max_transformers_version.
    """
    arch = _find_arch_with_max_version()
    if arch is None:
        pytest.skip("No model with max_transformers_version found in registry")

    info = HF_EXAMPLE_MODELS.get_hf_info(arch)
    original_version = reg.TRANSFORMERS_VERSION
    try:
        reg.TRANSFORMERS_VERSION = "99.0.0"
        result = info.check_transformers_version(
            on_fail="return",
            check_max_version=True,
        )
        assert result is not None, (
            f"check_transformers_version(check_max_version=True) returned None "
            f"for {arch!r} with transformers==99.0.0. "
            "Expected a non-None message indicating the model should be skipped."
        )
    finally:
        reg.TRANSFORMERS_VERSION = original_version


def test_test_initialization_uses_check_max_version_false():
    """Regression test: test_initialization.py must call check_transformers_version
    with check_max_version=False so models are not skipped with nightly transformers.
    """
    import os
    path = os.path.join(
        os.path.dirname(__file__), "test_initialization.py"
    )
    with open(path) as f:
        content = f.read()

    assert "check_max_version=False" in content, (
        "tests/models/test_initialization.py must call "
        "check_transformers_version(check_max_version=False) to avoid "
        "silently skipping models when running with nightly transformers."
    )


def test_test_transformers_uses_check_max_version_false():
    """Regression test: test_transformers.py get_model() must call
    check_transformers_version with check_max_version=False.
    """
    import os
    path = os.path.join(
        os.path.dirname(__file__), "test_transformers.py"
    )
    with open(path) as f:
        content = f.read()

    assert "check_max_version=False" in content, (
        "tests/models/test_transformers.py get_model() must call "
        "check_transformers_version(check_max_version=False) to avoid "
        "silently skipping models when running with nightly transformers."
    )
