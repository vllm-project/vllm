# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from vllm.platforms import _kfd_topology_has_amd_gpu


def _write_properties(node: Path, properties: dict[str, int]) -> None:
    node.mkdir(parents=True)
    (node / "properties").write_text(
        "".join(f"{key} {value}\n" for key, value in properties.items()),
        encoding="utf-8",
    )


def test_kfd_topology_detects_amd_gpu(tmp_path: Path):
    # CPU node: no GFX target.
    _write_properties(
        tmp_path / "0",
        {
            "vendor_id": 0,
            "device_id": 0,
            "gfx_target_version": 0,
        },
    )
    # GPU node: AMD vendor ID (0x1002) and non-zero GFX target.
    _write_properties(
        tmp_path / "1",
        {
            "vendor_id": 0x1002,
            "device_id": 0x1586,
            "gfx_target_version": 110501,
        },
    )

    assert _kfd_topology_has_amd_gpu(str(tmp_path))


def test_kfd_topology_rejects_missing_or_non_gpu_nodes(tmp_path: Path):
    assert not _kfd_topology_has_amd_gpu(str(tmp_path / "missing"))

    _write_properties(
        tmp_path / "0",
        {
            "vendor_id": 0,
            "device_id": 0,
            "gfx_target_version": 0,
        },
    )
    _write_properties(
        tmp_path / "1",
        {
            "vendor_id": 0x10DE,
            "device_id": 0x1234,
            "gfx_target_version": 110501,
        },
    )

    assert not _kfd_topology_has_amd_gpu(str(tmp_path))
