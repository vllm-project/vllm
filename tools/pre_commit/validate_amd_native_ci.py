#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from typing import Any

import yaml

TEST_AMD = Path(".buildkite/test-amd.yaml")
TEST_AREAS = Path(".buildkite/test_areas")


def _parse_amd_device(device: Any) -> tuple[str, int] | None:
    model, separator, gpu_count = str(device or "").rpartition("_")
    if (
        separator
        and model.startswith("mi")
        and model.removeprefix("mi").isdigit()
        and gpu_count.isdigit()
        and int(gpu_count) > 0
    ):
        return model, int(gpu_count)
    return None


def _first_configured(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _validate_runtime(
    *,
    label: str,
    device: Any,
    native: Any,
    num_gpus: Any,
    num_nodes: Any,
    no_plugin: Any,
    require_explicit_gpu_count: bool,
) -> list[str]:
    errors: list[str] = []
    parsed_device = _parse_amd_device(device)
    if parsed_device is None:
        return [f"{label}: invalid AMD device or agent pool {device!r}"]

    _, expected_gpus = parsed_device
    if not isinstance(native, bool):
        errors.append(f"{label}: native_ci must be a boolean")

    if native and (num_nodes or 1) > 1:
        errors.append(f"{label}: native AMD jobs cannot be multi-node")
    if native and no_plugin:
        errors.append(f"{label}: native AMD jobs cannot use no_plugin")

    configured_gpus = 1 if require_explicit_gpu_count and num_gpus is None else num_gpus
    if configured_gpus is not None and configured_gpus != expected_gpus:
        errors.append(
            f"{label}: GPU count {configured_gpus} does not match "
            f"{device} ({expected_gpus})"
        )

    return errors


def _load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def validate_test_amd_config() -> tuple[list[str], int]:
    errors: list[str] = []
    validated = 0
    for index, step in enumerate(_load(TEST_AMD).get("steps", []), start=1):
        if not isinstance(step, dict) or "mirror_hardwares" not in step:
            continue
        label = f"{TEST_AMD}:{step.get('label', f'step {index}')}"
        errors.extend(
            _validate_runtime(
                label=label,
                device=step.get("agent_pool"),
                native=step.get("native_ci", False),
                num_gpus=step.get("num_gpus"),
                num_nodes=step.get("num_nodes"),
                no_plugin=step.get("no_plugin", False),
                require_explicit_gpu_count=True,
            )
        )
        validated += 1
    return errors, validated


def validate_test_areas() -> tuple[list[str], int]:
    errors: list[str] = []
    validated = 0
    for path in sorted(TEST_AREAS.glob("*.yaml")):
        for index, step in enumerate(_load(path).get("steps", []), start=1):
            if not isinstance(step, dict):
                continue
            label = f"{path}:{step.get('label', f'step {index}')}"
            direct_device = step.get("device")
            direct_is_amd = _parse_amd_device(direct_device) is not None
            if direct_is_amd:
                errors.extend(
                    _validate_runtime(
                        label=label,
                        device=direct_device,
                        native=step.get("native_ci", False),
                        num_gpus=_first_configured(
                            step.get("num_devices"), step.get("num_gpus")
                        ),
                        num_nodes=step.get("num_nodes"),
                        no_plugin=step.get("no_plugin", False),
                        require_explicit_gpu_count=True,
                    )
                )
                validated += 1
            elif "native_ci" in step:
                errors.append(f"{label}: native_ci is set on a non-AMD direct step")

            mirror = step.get("mirror")
            amd = mirror.get("amd") if isinstance(mirror, dict) else None
            if not isinstance(amd, dict):
                continue
            errors.extend(
                _validate_runtime(
                    label=f"{label} [AMD mirror]",
                    device=amd.get("device"),
                    native=amd.get("native_ci", False),
                    num_gpus=_first_configured(
                        amd.get("num_devices"),
                        amd.get("num_gpus"),
                        step.get("num_devices"),
                        step.get("num_gpus"),
                    ),
                    num_nodes=amd.get("num_nodes", step.get("num_nodes")),
                    no_plugin=amd.get("no_plugin", step.get("no_plugin", False)),
                    require_explicit_gpu_count=True,
                )
            )
            validated += 1
    return errors, validated


def main() -> int:
    test_amd_errors, test_amd_count = validate_test_amd_config()
    area_errors, area_count = validate_test_areas()
    errors = test_amd_errors + area_errors
    if errors:
        raise SystemExit(
            "Invalid AMD native CI configuration:\n- " + "\n- ".join(errors)
        )

    print(
        "AMD native CI configuration validation passed "
        f"({test_amd_count} test-amd steps, {area_count} test-area steps)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
