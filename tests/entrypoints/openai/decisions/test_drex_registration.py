# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registration tests for the Drex architecture.

These import the model package, which needs the engine's torch build, so they
run where that is installed (the GPU box); elsewhere the module is skipped.
"""

import pytest

pytest.importorskip("vllm.model_executor.models.config")

from vllm.model_executor.models.config import MODELS_CONFIG_MAP  # noqa: E402
from vllm.model_executor.models.registry import ModelRegistry  # noqa: E402


def test_config_map_routes_drex():
    from vllm.model_executor.models.config import Qwen3_5ForDrexConfig

    assert MODELS_CONFIG_MAP["Qwen3_5ForDrex"] is Qwen3_5ForDrexConfig


def test_registry_resolves_drex():
    from vllm.model_executor.models.qwen3_5_drex import Qwen3_5ForDrex

    assert ModelRegistry.models["Qwen3_5ForDrex"].load_model_cls() is Qwen3_5ForDrex
