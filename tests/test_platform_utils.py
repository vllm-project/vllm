# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from unittest.mock import Mock

import pytest

import vllm
from vllm.utils.platform_utils import resolve_rocm_device_config_file_path


def _config_path(directory: Path, device_name: str, *, experts: int = 128) -> Path:
    return directory / (
        f"E={experts},N=768,device_name={device_name},dtype=int4_w4a16.json"
    )


@pytest.mark.parametrize(
    "requested_name", ["AMD_Radeon_8060S", "AMD_Radeon_8060S_Graphics"]
)
def test_resolve_device_config_file_path_uses_equivalent_device_name(
    monkeypatch,
    tmp_path: Path,
    requested_name: str,
) -> None:
    monkeypatch.setattr("vllm.platforms.current_platform.is_rocm", lambda: True)
    requested = _config_path(tmp_path, requested_name)
    existing = _config_path(tmp_path, "Radeon_8060S_Graphics")
    existing.write_text("{}")

    assert resolve_rocm_device_config_file_path(str(requested)) == str(existing)


def test_resolve_device_config_file_path_finds_shipped_gfx1151_config(
    monkeypatch,
) -> None:
    monkeypatch.setattr("vllm.platforms.current_platform.is_rocm", lambda: True)
    config_dir = (
        Path(vllm.__file__).parent
        / "model_executor"
        / "layers"
        / "fused_moe"
        / "configs"
    )
    requested = _config_path(config_dir, "AMD_Radeon_8060S")
    existing = _config_path(config_dir, "Radeon_8060S_Graphics")

    assert existing.is_file()
    assert resolve_rocm_device_config_file_path(str(requested)) == str(existing)


def test_resolve_device_config_file_path_prefers_exact_name(
    monkeypatch, tmp_path: Path
) -> None:
    is_rocm = Mock(return_value=True)
    monkeypatch.setattr("vllm.platforms.current_platform.is_rocm", is_rocm)
    requested = _config_path(tmp_path, "AMD_Radeon_8060S")
    equivalent = _config_path(tmp_path, "Radeon_8060S_Graphics")
    requested.write_text("exact")
    equivalent.write_text("equivalent")

    assert resolve_rocm_device_config_file_path(str(requested)) == str(requested)
    is_rocm.assert_not_called()


def test_resolve_device_config_file_path_preserves_other_selectors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("vllm.platforms.current_platform.is_rocm", lambda: True)
    requested = _config_path(tmp_path, "AMD_Radeon_8060S")
    _config_path(tmp_path, "Radeon_8060S_Graphics", experts=256).write_text("{}")

    assert resolve_rocm_device_config_file_path(str(requested)) == str(requested)


def test_resolve_device_config_file_path_rejects_ambiguous_aliases(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("vllm.platforms.current_platform.is_rocm", lambda: True)
    requested = _config_path(tmp_path, "AMD_Radeon_8060S")
    _config_path(tmp_path, "Radeon_8060S_Graphics").write_text("first")
    _config_path(tmp_path, "AMD-Radeon-8060S-Graphics").write_text("second")

    assert resolve_rocm_device_config_file_path(str(requested)) == str(requested)


def test_resolve_device_config_file_path_rejects_empty_device_name(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("vllm.platforms.current_platform.is_rocm", lambda: True)
    requested = _config_path(tmp_path, "AMD_Graphics")
    _config_path(tmp_path, "Graphics").write_text("{}")

    assert resolve_rocm_device_config_file_path(str(requested)) == str(requested)


def test_resolve_device_config_file_path_does_not_alias_non_rocm(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("vllm.platforms.current_platform.is_rocm", lambda: False)
    requested = _config_path(tmp_path, "AMD_Radeon_8060S")
    _config_path(tmp_path, "Radeon_8060S_Graphics").write_text("{}")

    assert resolve_rocm_device_config_file_path(str(requested)) == str(requested)
