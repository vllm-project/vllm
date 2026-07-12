# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from typing import Any

import pytest
import yaml

from tools.pre_commit import validate_amd_native_ci as validator


@pytest.fixture
def config_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    test_amd = tmp_path / "test-amd.yaml"
    test_areas = tmp_path / "test_areas"
    test_areas.mkdir()
    monkeypatch.setattr(validator, "TEST_AMD", test_amd)
    monkeypatch.setattr(validator, "TEST_AREAS", test_areas)
    return test_amd, test_areas


def _write_yaml(path: Path, steps: list[dict[str, Any]]) -> None:
    path.write_text(yaml.safe_dump({"steps": steps}, sort_keys=False))


def _legacy_step(
    device: str,
    *,
    native_ci: Any = None,
    **overrides: Any,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "label": f"Legacy {device}",
        "mirror_hardwares": ["amdproduction"],
        "agent_pool": device,
    }
    if native_ci is not None:
        step["native_ci"] = native_ci
    step.update(overrides)
    return step


def test_legacy_accepts_explicit_modes_for_current_and_future_devices(
    config_paths,
) -> None:
    test_amd, _ = config_paths
    steps = [
        _legacy_step("mi300_1", native_ci=True),
        _legacy_step("mi300_1", native_ci=False),
        _legacy_step("mi325_1", native_ci=True),
        _legacy_step("mi325_1"),
        _legacy_step("mi400_16", native_ci=True, num_gpus=16),
    ]
    _write_yaml(test_amd, steps)

    errors, count = validator.validate_test_amd_config()

    assert errors == []
    assert count == 5


@pytest.mark.parametrize(
    "step",
    [
        _legacy_step("mi300_1", native_ci="true"),
        _legacy_step("mi400_16", native_ci=1, num_gpus=16),
    ],
)
def test_legacy_rejects_non_boolean_native_flag(
    config_paths,
    step: dict[str, Any],
) -> None:
    test_amd, _ = config_paths
    _write_yaml(test_amd, [step])

    errors, _ = validator.validate_test_amd_config()

    assert any("native_ci must be a boolean" in error for error in errors)


@pytest.mark.parametrize(
    "step",
    [
        _legacy_step("mi400_16", native_ci=True),
        _legacy_step("mi400_16", native_ci=True, num_gpus=8),
    ],
)
def test_legacy_rejects_gpu_count_that_does_not_match_pool_suffix(
    config_paths,
    step: dict[str, Any],
) -> None:
    test_amd, _ = config_paths
    _write_yaml(test_amd, [step])

    errors, _ = validator.validate_test_amd_config()

    assert any("does not match mi400_16 (16)" in error for error in errors)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_nodes": 2}, "cannot be multi-node"),
        ({"no_plugin": True}, "cannot use no_plugin"),
    ],
)
def test_legacy_rejects_unsupported_native_modes(
    config_paths,
    overrides: dict[str, Any],
    message: str,
) -> None:
    test_amd, _ = config_paths
    step = _legacy_step("mi400_16", native_ci=True, num_gpus=16, **overrides)
    _write_yaml(test_amd, [step])

    errors, _ = validator.validate_test_amd_config()

    assert any(message in error for error in errors)


def test_test_areas_accept_direct_and_nested_amd_devices(config_paths) -> None:
    _, test_areas = config_paths
    _write_yaml(
        test_areas / "valid.yaml",
        [
            {
                "label": "Direct native future device",
                "device": "mi400_2",
                "native_ci": True,
                "num_devices": 2,
            },
            {
                "label": "Direct legacy current device",
                "device": "mi300_1",
            },
            {
                "label": "Label does not determine runtime mode",
                "device": "h100",
                "mirror": {"amd": {"device": "mi300_1"}},
            },
            {
                "label": "Nested native mirror",
                "device": "h100",
                "num_devices": 16,
                "mirror": {
                    "amd": {
                        "device": "mi400_16",
                        "native_ci": True,
                    }
                },
            },
        ],
    )

    errors, count = validator.validate_test_areas()

    assert errors == []
    assert count == 4


def test_test_areas_reject_native_flag_on_non_amd_direct_step(config_paths) -> None:
    _, test_areas = config_paths
    _write_yaml(
        test_areas / "invalid.yaml",
        [{"label": "Not actually AMD", "device": "h100", "native_ci": True}],
    )

    errors, count = validator.validate_test_areas()

    assert count == 0
    assert any("native_ci is set on a non-AMD direct step" in error for error in errors)


@pytest.mark.parametrize(
    ("amd", "message"),
    [
        ({"device": "mi400_1", "native_ci": "true"}, "must be a boolean"),
        (
            {"device": "mi400_1", "native_ci": True, "num_nodes": 2},
            "cannot be multi-node",
        ),
        (
            {"device": "mi400_1", "native_ci": True, "no_plugin": True},
            "cannot use no_plugin",
        ),
    ],
)
def test_test_areas_reject_invalid_nested_amd_policy(
    config_paths,
    amd: dict[str, Any],
    message: str,
) -> None:
    _, test_areas = config_paths
    _write_yaml(
        test_areas / "invalid.yaml",
        [{"label": "Nested mirror", "device": "h100", "mirror": {"amd": amd}}],
    )

    errors, _ = validator.validate_test_areas()

    assert any(message in error for error in errors)


def test_mirror_gpu_count_inherits_top_level_count_for_validation(config_paths) -> None:
    _, test_areas = config_paths
    _write_yaml(
        test_areas / "invalid.yaml",
        [
            {
                "label": "Mismatched mirror GPU count",
                "device": "h100",
                "num_devices": 8,
                "mirror": {
                    "amd": {
                        "device": "mi300_4",
                        "native_ci": True,
                    }
                },
            }
        ],
    )

    errors, _ = validator.validate_test_areas()

    assert any("GPU count 8 does not match mi300_4 (4)" in error for error in errors)
