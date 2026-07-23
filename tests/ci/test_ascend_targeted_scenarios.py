# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/ascend_targeted_scenarios.py"
)
REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/ascend_targeted_scenarios.json"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "ascend_targeted_scenarios", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_targeted_scenario_registry_from_repo_file():
    module = load_module()

    registry = module.load_targeted_scenario_registry(REGISTRY_PATH)

    assert registry.resolve_scenario("random") == "random-online"
    assert registry.resolve_scenario("sharegpt-online") == "sharegpt-online"
    assert registry.resolve_group("smoke") == "random-online"
    assert registry.resolve_group("sharegpt") == "sharegpt-online"
    assert registry.supported_scenarios == ["random-online", "sharegpt-online"]
    assert registry.supported_groups == ["random", "sharegpt", "smoke"]


def test_registry_rejects_unknown_group_scenario(tmp_path):
    module = load_module()
    registry_path = tmp_path / "bad-registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "scenarios": {
                    "random-online": {"aliases": ["random"]},
                },
                "groups": {
                    "moe": {"scenario": "moe-online"},
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        module.load_targeted_scenario_registry(registry_path)
    except ValueError as exc:
        assert "unknown scenario" in str(exc)
    else:
        raise AssertionError("expected unknown group scenario to be rejected")


def test_registry_rejects_duplicate_alias(tmp_path):
    module = load_module()
    registry_path = tmp_path / "bad-registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "scenarios": {
                    "random-online": {"aliases": ["preview"]},
                    "sharegpt-online": {"aliases": ["preview"]},
                },
                "groups": {},
            }
        ),
        encoding="utf-8",
    )

    try:
        module.load_targeted_scenario_registry(registry_path)
    except ValueError as exc:
        assert "maps to both" in str(exc)
    else:
        raise AssertionError("expected duplicate alias to be rejected")
