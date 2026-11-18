# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm device queries that must work without the usual source: amdgpu's exact
VRAM counter from KFD topology and DRM sysfs (plain files in a temporary
directory), and RocmPlatform's fallbacks when amdsmi cannot initialize, as on
WSL2. Both run without a GPU."""

import importlib.util
import types

import pytest
import torch

from vllm.platforms.rocm_device_memory import find_vram_used_file

GB = 1 << 30
# torch's ROCm UUID holds the KFD unique_id 0x13a691b49a73a35c as ASCII hex.
UUID = b"13a691b49a73a35c".hex()
UNIQUE_ID = 0x13A691B49A73A35C
# location_id 0xE300 is PCI bus 0xE3, device 0, function 0.
PCI = (0, 0xE3, 0)


def _node(root, name, **props):
    node = root / "nodes" / name
    node.mkdir(parents=True)
    node.joinpath("properties").write_text(
        "".join(f"{k} {v}\n" for k, v in props.items())
    )


def _gpu(root, name="2", minor=128, location_id=0xE300, unique_id=UNIQUE_ID):
    _node(
        root,
        name,
        simd_count=128,
        drm_render_minor=minor,
        location_id=location_id,
        domain=0,
        unique_id=unique_id,
    )


def _vram(root, minor=128, total=32 * GB, used=GB):
    device = root / "drm" / f"renderD{minor}" / "device"
    device.mkdir(parents=True)
    device.joinpath("mem_info_vram_total").write_text(f"{total}\n")
    device.joinpath("mem_info_vram_used").write_text(f"{used}\n")
    return str(device / "mem_info_vram_used")


def _find(root, uuid=UUID, pci=PCI, total=32 * GB):
    return find_vram_used_file(
        uuid,
        pci,
        total,
        nodes_dir=str(root / "nodes"),
        drm_dir=str(root / "drm"),
    )


@pytest.fixture
def root(tmp_path):
    # A CPU node, skipped by the lookup.
    _node(tmp_path, "0", simd_count=0, drm_render_minor=0, location_id=0)
    return tmp_path


def test_finds_the_gpu_by_uuid_and_pci(root):
    _gpu(root)
    _gpu(root, name="3", minor=129, location_id=0x6300, unique_id=7)
    expected = _vram(root)
    _vram(root, minor=129)
    assert _find(root) == (expected, "")
    # A UUID that is not hex digits leaves the PCI address.
    assert _find(root, uuid="GPU-1234") == (expected, "")


@pytest.mark.parametrize(
    ("uuid", "pci"),
    [
        (UUID, (0, 0x99, 0)),  # no node at this PCI address
        (b"0000000000000007".hex(), PCI),  # the node at this address has
        # another unique_id
    ],
)
def test_no_matching_node(root, uuid, pci):
    _gpu(root)
    _vram(root)
    path, reason = _find(root, uuid=uuid, pci=pci)
    assert path is None and "no single KFD node" in reason


def test_skips_partitioned_gpus(root):
    """Partitions of one GPU share its PCI bus and device (function 1 here)
    and its VRAM counter."""
    _gpu(root)
    _gpu(root, name="3", minor=129, location_id=0xE301, unique_id=7)
    _vram(root)
    _vram(root, minor=129)
    assert _find(root) == (None, "partitioned GPU")


def test_skips_a_malformed_node(root):
    """A node whose properties do not parse is skipped, not a crash."""
    _gpu(root, unique_id="not-a-number")
    _vram(root)
    path, reason = _find(root)
    assert path is None and "no single KFD node" in reason


def test_needs_the_vram_counters(root):
    _gpu(root)
    path, reason = _find(root)
    assert path is None and "no amdgpu VRAM counters" in reason


@pytest.mark.parametrize(
    ("vram_total", "found"),
    [
        (32 * GB, True),
        (int(32 * GB * 1.005), True),
        # More than 1% above HIP's total: HIP's memory is not this VRAM.
        (int(32 * GB * 1.02), False),
        # Below HIP's total (e.g. an APU's small carve-out).
        (32 * GB - (1 << 20), False),
    ],
)
def test_vram_total_must_match_hip(root, vram_total, found):
    _gpu(root)
    expected = _vram(root, total=vram_total)
    path, reason = _find(root)
    assert path == (expected if found else None)
    assert found or "does not match" in reason


def test_no_kfd_topology(tmp_path):
    path, reason = _find(tmp_path)
    assert path is None and "no KFD topology" in reason


GPU_NAME = "AMD Radeon RX 9060 XT"


class _DeviceProperties:
    gcnArchName = "gfx1200"


def _amdsmi_init_fails() -> None:
    # What amdsmi raises on WSL2 (https://github.com/vllm-project/vllm/issues/64664).
    raise RuntimeError("AmdSmiLibraryException(34)")


@pytest.fixture
def rocm(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    # Private copy, because the import-time GPU arch query is stubbed and must not
    # leak into sys.modules for other tests.
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda *_: _DeviceProperties()
    )
    spec = importlib.util.find_spec("vllm.platforms.rocm")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # raising=False because amdsmi_init does not exist on hosts without amdsmi.
    monkeypatch.setattr(module, "amdsmi_init", _amdsmi_init_fails, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: GPU_NAME)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return module


def test_get_device_name_falls_back_to_torch(rocm: types.ModuleType) -> None:
    assert rocm.RocmPlatform.get_device_name(0) == GPU_NAME


@pytest.mark.parametrize("init_fails", [True, False])
@pytest.mark.parametrize(
    ("visible_devices", "expected"), [(None, "rocm:0"), ("1", "rocm:1")]
)
def test_get_device_uuid_falls_back_to_physical_id(
    rocm: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    init_fails: bool,
    visible_devices: str | None,
    expected: str,
) -> None:
    """Also when amdsmi starts but the UUID query fails: a per-GPU placeholder,
    so callers matching GPUs by UUID never see every GPU as the same one."""
    if not init_fails:

        def uuid_query_fails(_handle):
            # AmdSmiException where amdsmi is installed.
            raise getattr(rocm, "AmdSmiException", RuntimeError)("uuid")

        monkeypatch.setattr(rocm, "amdsmi_init", lambda: None, raising=False)
        monkeypatch.setattr(rocm, "amdsmi_shut_down", lambda: None, raising=False)
        monkeypatch.setattr(
            rocm, "amdsmi_get_processor_handles", lambda: [0, 1], raising=False
        )
        monkeypatch.setattr(
            rocm, "amdsmi_get_gpu_device_uuid", uuid_query_fails, raising=False
        )
    if visible_devices is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    assert rocm.RocmPlatform.get_device_uuid(0) == expected


def test_is_fully_connected_is_false_without_amdsmi(rocm: types.ModuleType) -> None:
    assert not rocm.RocmPlatform.is_fully_connected([0, 1])


def test_numa_nodes_are_unknown_without_amdsmi(
    rocm: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rocm.RocmPlatform, "device_count", classmethod(lambda _: 2))
    assert rocm.RocmPlatform.get_all_device_numa_nodes() is None
