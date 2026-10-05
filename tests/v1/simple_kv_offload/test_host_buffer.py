# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HostBuffer for SimpleCPUOffloadConnector (CPU only, Linux)."""

import mmap
import sys

import pytest

from vllm.v1.simple_kv_offload.host_buffer import HostBuffer, parse_hugepage_size

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux mm APIs")


def free_hugepages(page: int) -> int:
    path = f"/sys/kernel/mm/hugepages/hugepages-{page >> 10}kB/free_hugepages"
    try:
        with open(path) as f:
            return int(f.read())
    except OSError:
        return -1


def hugepage_with_free(n: int) -> int:
    for page in (2 << 20, 1 << 30):
        if free_hugepages(page) >= n:
            return page
    pytest.skip(f"needs {n} free hugepages of some size")
    raise AssertionError("unreachable")


def mapping_info(addr: int) -> tuple[set[str], str]:
    """(smaps VmFlags, numa_maps policy) of the mapping starting at ``addr``."""
    flags: set[str] = set()
    with open("/proc/self/smaps") as f:
        inside = False
        for line in f:
            head = line.split()[0]
            if "-" in head and not head.endswith(":"):
                inside = int(head.split("-")[0], 16) == addr
            elif inside and line.startswith("VmFlags:"):
                flags = set(line.split()[1:])
                break
    with open("/proc/self/numa_maps") as f:
        policy = next(ln.split()[1] for ln in f if int(ln.split()[0], 16) == addr)
    return flags, policy


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("none", None),
        ("2MB", 2 << 20),
        ("1GiB", 1 << 30),
        ("1g", 1 << 30),
    ],
)
def test_parse_hugepage_size(value, expected):
    assert parse_hugepage_size(value) == expected


def test_parse_hugepage_size_rejects_unknown():
    with pytest.raises(ValueError, match="cpu_hugepage_block_size"):
        parse_hugepage_size("4KB")


@pytest.mark.parametrize("hugepages", [False, True])
def test_buffer_rounds_up_populates_and_skips_fork(hugepages: bool):
    page = hugepage_with_free(2) if hugepages else mmap.PAGESIZE
    buf = HostBuffer(page + 1, page if hugepages else None, numa_node=0)
    assert buf.nbytes == 2 * page
    assert buf.tensor.data_ptr() % page == 0
    assert int(buf.tensor[::4096].abs().sum()) == 0  # kernel-zeroed
    flags, policy = mapping_info(buf.tensor.data_ptr())
    assert "dc" in flags  # MADV_DONTFORK sets VM_DONTCOPY
    assert ("ht" in flags) == hugepages
    assert policy == "prefer:0"


def test_hugepage_buffer_fails_closed_when_pool_is_short():
    page = 1 << 30
    free = free_hugepages(page)
    if free < 0:
        pytest.skip("no 1 GiB hugepage pool")
    with pytest.raises(RuntimeError, match="Cannot map"):
        HostBuffer((free + 1) * page, page)
