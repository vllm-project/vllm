# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import json
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = REPO_ROOT / "vllm"
MAPPING_PATH = REPO_ROOT / "tools" / "package_refactor" / "mapping.json"


@pytest.mark.parametrize(
    ("legacy_path", "canonical_path"),
    [
        (
            "vllm.connections",
            "vllm.foundation.system.connections",
        ),
        (
            "vllm.sampling_params",
            "vllm.frontend.processing.sampling_params",
        ),
        (
            "vllm.logprobs",
            "vllm.runtime.generation.logprobs",
        ),
        (
            "vllm.scalar_type",
            "vllm.backends.compute.scalar_type",
        ),
        (
            "vllm.triton_utils.force_first_config",
            "vllm.backends.compute.dsl.triton_utils.force_first_config",
        ),
    ],
)
def test_legacy_leaf_modules_alias_canonical_modules(
    legacy_path: str, canonical_path: str
) -> None:
    legacy = importlib.import_module(legacy_path)
    canonical = importlib.import_module(canonical_path)

    assert legacy is canonical


@pytest.mark.parametrize(
    ("legacy_path", "canonical_path", "symbol"),
    [
        ("vllm.config", "vllm.foundation.config", "VllmConfig"),
        ("vllm.inputs", "vllm.frontend.processing.inputs", "PromptType"),
        ("vllm.platforms", "vllm.backends.platform", "Platform"),
    ],
)
def test_legacy_packages_export_canonical_symbols(
    legacy_path: str, canonical_path: str, symbol: str
) -> None:
    legacy = importlib.import_module(legacy_path)
    canonical = importlib.import_module(canonical_path)

    assert getattr(legacy, symbol) is getattr(canonical, symbol)


def test_public_root_api_uses_canonical_objects() -> None:
    import vllm

    entrypoints = importlib.import_module("vllm.frontend.entrypoints.llm")
    processing = importlib.import_module("vllm.frontend.processing.sampling_params")

    assert vllm.LLM is entrypoints.LLM
    assert vllm.SamplingParams is processing.SamplingParams


def test_cli_and_builtin_plugin_entrypoints_use_canonical_paths() -> None:
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert (
        pyproject["project"]["scripts"]["vllm"]
        == "vllm.frontend.entrypoints.cli.main:main"
    )
    plugin_paths = pyproject["project"]["entry-points"]["vllm.general_plugins"]
    assert plugin_paths
    assert all(
        path.startswith("vllm.foundation.extensibility.plugins.")
        for path in plugin_paths.values()
    )


def test_representative_dynamic_registries_remain_available() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    assert "Qwen3ForCausalLM" in ModelRegistry.get_supported_archs()
    assert AttentionBackendEnum.FLASH_ATTN.get_path().startswith("vllm.v1.")


def test_mapping_manifest_covers_existing_legacy_and_canonical_paths() -> None:
    moves = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))

    assert len(moves) == 42
    for move in moves:
        legacy = PACKAGE_ROOT / move["old"]
        canonical = PACKAGE_ROOT / move["new"]
        assert legacy.exists(), move
        assert canonical.exists(), move
