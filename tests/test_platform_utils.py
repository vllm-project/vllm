# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

import vllm
from vllm.utils.platform_utils import (
    get_device_name_as_file_name,
    resolve_rocm_device_config_file_path,
)


def _config_path(directory: Path, device_name: str, *, experts: int = 128) -> Path:
    return directory / (
        f"E={experts},N=768,device_name={device_name},dtype=int4_w4a16.json"
    )


@pytest.mark.parametrize(
    ("torch_name", "expected_file_name"),
    [
        ("Radeon 8060S Graphics", "Radeon_8060S_Graphics"),
        ("AMD Radeon 8060S Graphics", "AMD_Radeon_8060S_Graphics"),
    ],
)
def test_magicmock_device_name_falls_back_to_torch_and_resolves_config(
    monkeypatch,
    tmp_path: Path,
    torch_name: str,
    expected_file_name: str,
) -> None:
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform, "get_device_name", Mock(return_value=MagicMock())
    )
    monkeypatch.setattr(current_platform, "is_rocm", Mock(return_value=True))
    torch_get_device_name = Mock(return_value=torch_name)
    monkeypatch.setattr(
        "vllm.utils.platform_utils.torch.cuda.get_device_name",
        torch_get_device_name,
    )
    existing = _config_path(tmp_path, "Radeon_8060S_Graphics")
    existing.write_text("{}")
    get_device_name_as_file_name.cache_clear()

    try:
        requested_name = get_device_name_as_file_name()
        requested = _config_path(tmp_path, requested_name)
        assert requested_name == expected_file_name
        assert resolve_rocm_device_config_file_path(str(requested)) == str(existing)
    finally:
        get_device_name_as_file_name.cache_clear()

    torch_get_device_name.assert_called_once_with(0)


def test_get_device_name_as_file_name_keeps_valid_platform_name(monkeypatch) -> None:
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform, "get_device_name", Mock(return_value="AMD_Radeon_8060S")
    )
    is_rocm = Mock(return_value=True)
    monkeypatch.setattr(current_platform, "is_rocm", is_rocm)
    torch_get_device_name = Mock()
    monkeypatch.setattr(
        "vllm.utils.platform_utils.torch.cuda.get_device_name",
        torch_get_device_name,
    )
    get_device_name_as_file_name.cache_clear()

    try:
        assert get_device_name_as_file_name() == "AMD_Radeon_8060S"
    finally:
        get_device_name_as_file_name.cache_clear()

    is_rocm.assert_not_called()
    torch_get_device_name.assert_not_called()


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
