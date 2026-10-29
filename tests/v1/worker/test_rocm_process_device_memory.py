# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parsing of DRM fdinfo and KFD topology behind
RocmPlatform.get_process_device_memory_bytes(). The inputs
are text in the format the amdgpu driver prints, but importing
vllm.platforms.rocm needs a ROCm GPU."""

import mmap

import pytest

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    # vllm.platforms.rocm queries the GPU at import time.
    pytest.skip("requires ROCm", allow_module_level=True)

from vllm.platforms import rocm  # noqa: E402

# Shortened /proc/self/fdinfo/<fd> of an amdgpu render node (Linux 7.0).
# Real output; pasid and the GTT/resident keys are not used by the parser.
FDINFO_NEW_KERNEL = """\
pos:\t0
flags:\t02100002
drm-driver:\tamdgpu
drm-client-id:\t122
drm-pdev:\t0000:e3:00.0
pasid:\t117
drm-total-gtt:\t2128 KiB
drm-total-vram:\t1052672 KiB
drm-resident-vram:\t1052672 KiB
drm-memory-vram:\t1052672 KiB
amd-requested-vram:\t1052672 KiB
"""
# Kernels before drm-total-*: only the amdgpu-specific key.
FDINFO_OLD_KERNEL = (
    "drm-driver:\tamdgpu\ndrm-client-id:\t7\namd-requested-vram:\t4096\n"
)
# MI300 partitions 1..N-1 print no memory keys at all.
FDINFO_NO_MEMORY = "drm-driver:\tamdgpu\ndrm-client-id:\t9\n"
# Real VRAM/GTT lines of a ROCm process on an R9700 (Linux 7.0.0), render node
# 132, after a 1 GiB tensor. All VRAM keys are equal: nothing is evicted.
FDINFO_LINUX_7_0 = """\
drm-driver:\tamdgpu
drm-client-id:\t35220
pasid:\t35214
drm-total-gtt:\t8280 KiB
drm-shared-gtt:\t0
drm-resident-gtt:\t8240 KiB
drm-purgeable-gtt:\t0
drm-total-vram:\t1193664 KiB
drm-shared-vram:\t0
drm-resident-vram:\t1193664 KiB
drm-purgeable-vram:\t0
drm-memory-vram:\t1193664 KiB
drm-memory-gtt:\t8240 KiB
amd-evicted-vram:\t0 KiB
amd-requested-vram:\t1193664 KiB
amd-requested-gtt:\t8280 KiB
"""
# Kernels that print only the deprecated alias of drm-resident-vram.
FDINFO_RESIDENT_ONLY = (
    "drm-driver:\tamdgpu\ndrm-client-id:\t11\ndrm-memory-vram:\t4096 KiB\n"
)


def test_sum_fdinfo_vram_reads_the_vram_key():
    assert rocm._sum_fdinfo_vram([FDINFO_NEW_KERNEL]) == 1052672 << 10
    assert rocm._sum_fdinfo_vram([FDINFO_OLD_KERNEL]) == 4096


def test_sum_fdinfo_vram_reads_total_from_linux_7_0_output():
    assert rocm._sum_fdinfo_vram([FDINFO_LINUX_7_0]) == 1193664 << 10
    # The same output under VRAM pressure: 256 MiB moved to GTT. The total
    # still counts it (it returns to VRAM when used); resident does not.
    evicted = (
        FDINFO_LINUX_7_0.replace(
            "drm-resident-vram:\t1193664", "drm-resident-vram:\t931520"
        )
        .replace("drm-memory-vram:\t1193664", "drm-memory-vram:\t931520")
        .replace("amd-evicted-vram:\t0", "amd-evicted-vram:\t262144")
    )
    assert rocm._sum_fdinfo_vram([evicted]) == 1193664 << 10


def test_sum_fdinfo_vram_ignores_the_resident_alias():
    # drm-memory-vram means resident, not total: not a usable reading.
    assert rocm._sum_fdinfo_vram([FDINFO_RESIDENT_ONLY]) is None


def test_sum_fdinfo_vram_without_memory_keys_is_none():
    assert rocm._sum_fdinfo_vram([FDINFO_NO_MEMORY]) is None
    assert rocm._sum_fdinfo_vram([]) is None


def test_sum_fdinfo_vram_counts_each_client_once():
    # A duplicated fd (same drm-client-id) is the same open file: count once.
    # A second open of the render node is a separate client: add it.
    texts = [FDINFO_NEW_KERNEL, FDINFO_NEW_KERNEL, FDINFO_OLD_KERNEL]
    assert rocm._sum_fdinfo_vram(texts) == (1052672 << 10) + 4096


def _write_node(root, name, **props):
    node = root / name
    node.mkdir()
    node.joinpath("properties").write_text(
        "".join(f"{k} {v}\n" for k, v in props.items())
    )


def test_kfd_gpu_node_matches_uuid_then_pci(tmp_path, monkeypatch):
    monkeypatch.setattr(rocm, "_KFD_TOPOLOGY_NODES", str(tmp_path))
    rocm._kfd_gpu_node_props.cache_clear()
    _write_node(tmp_path, "0", simd_count=0)  # CPU node, skipped
    # A GPU node. unique_id and location_id are independent properties;
    # location_id 0xE300 is PCI bus 0xE3, device 0, function 0.
    _write_node(
        tmp_path,
        "6",
        simd_count=64,
        drm_render_minor=132,
        location_id=0xE300,
        domain=0,
        unique_id=0x13A691B49A73A35C,
    )
    # torch writes the unique_id as ASCII hex characters in the UUID bytes.
    uuid = b"13a691b49a73a35c".hex()
    for key in (uuid, (0, 0x99, 0)), ("not-hex", (0, 0xE3, 0)):
        node = rocm._kfd_gpu_node_props(*key)
        assert node is not None and node["drm_render_minor"] == "132"
    # Same bus and device, function 1. The lookup ignores the function (torch
    # does not report it), so two nodes now match (0, 0xE3, 0); with no UUID
    # match the GPU is ambiguous.
    _write_node(tmp_path, "7", simd_count=64, drm_render_minor=133, location_id=0xE301)
    rocm._kfd_gpu_node_props.cache_clear()
    assert rocm._kfd_gpu_node_props("not-hex", (0, 0xE3, 0)) is None
    rocm._kfd_gpu_node_props.cache_clear()


# KFD topology properties that size each hardware queue's context save area.
# cwsr_size is a multiple of every page size, so no rounding applies.
QUEUE_PROPS = {"cwsr_size": str(10 * 65536), "num_xcc": "2"}
QUEUE_SAVE_AREA = 2 * 10 * 65536


def test_queue_save_area_is_cwsr_size_per_xcc_page_aligned():
    assert rocm._queue_save_area_bytes(QUEUE_PROPS) == QUEUE_SAVE_AREA
    # Rounded up to the page size; num_xcc is absent on older kernels.
    assert rocm._queue_save_area_bytes({"cwsr_size": "1"}) == mmap.PAGESIZE


def test_process_memory_adds_save_areas_for_the_queue_limit(monkeypatch):
    platform = rocm.RocmPlatform
    node = {"drm_render_minor": "128", **QUEUE_PROPS}
    monkeypatch.setattr(rocm, "_kfd_gpu_node_props", lambda uuid, pci: node)
    monkeypatch.setattr(rocm, "_sum_fdinfo_vram", lambda texts: 1000)
    monkeypatch.delenv("GPU_MAX_HW_QUEUES", raising=False)
    # HIP's default of 4 queues plus two runtime-internal ones.
    assert platform.get_process_device_memory_bytes("cuda:0") == (
        1000 + 6 * QUEUE_SAVE_AREA
    )
    monkeypatch.setenv("GPU_MAX_HW_QUEUES", "2")
    assert platform.get_process_device_memory_bytes("cuda:0") == (
        1000 + 4 * QUEUE_SAVE_AREA
    )


@pytest.mark.parametrize(
    "node, fdinfo_vram",
    [
        (None, 1000),  # no single KFD node for this GPU
        ({"drm_render_minor": "128"}, 1000),  # kernel without cwsr_size
        ({"drm_render_minor": "128", **QUEUE_PROPS}, None),  # no fdinfo VRAM key
    ],
)
def test_process_memory_falls_back_when_a_reading_is_missing(
    monkeypatch, node, fdinfo_vram
):
    monkeypatch.setattr(rocm, "_kfd_gpu_node_props", lambda uuid, pci: node)
    monkeypatch.setattr(rocm, "_sum_fdinfo_vram", lambda texts: fdinfo_vram)
    assert rocm.RocmPlatform.get_process_device_memory_bytes("cuda:0") is None
