# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts/ensure_vllm_provider.py"


def load_provider_helper():
    spec = importlib.util.spec_from_file_location("ensure_vllm_provider", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_distribution_defaults_to_current_pyproject_name():
    helper = load_provider_helper()

    assert helper.pyproject_distribution_name() == "vllm"
    assert helper.resolve_expected_distribution(None, ["vllm"]) == "vllm"


def test_current_pyproject_name_overrides_stale_requested_distribution():
    helper = load_provider_helper()

    assert (
        helper.resolve_expected_distribution("vllm-hust", ["vllm", "vllm-hust"])
        == "vllm"
    )


def test_current_pyproject_name_is_required_even_when_missing_from_providers():
    helper = load_provider_helper()

    assert helper.resolve_expected_distribution("vllm-hust", ["vllm-hust"]) == "vllm"
