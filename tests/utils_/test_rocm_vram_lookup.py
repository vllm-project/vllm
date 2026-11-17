# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Finding amdgpu's exact VRAM counter from KFD topology and DRM sysfs. Plain
files in a temporary directory, so this runs without a GPU."""

import pytest

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
