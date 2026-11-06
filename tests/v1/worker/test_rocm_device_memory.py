# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Free memory by the exact device memory count on a real ROCm GPU. The sysfs
lookup is tested without a GPU in tests/utils_/test_rocm_vram_lookup.py."""

import ctypes
import subprocess
import sys
import time

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    # vllm.platforms.rocm queries the GPU at import time.
    pytest.skip("requires ROCm", allow_module_level=True)

from vllm.platforms import rocm_device_memory  # noqa: E402
from vllm.utils.extensible_tensor import ExtensibleTensor  # noqa: E402

MB = 1 << 20
DEVICE = torch.device("cuda:0")


def _used() -> int:
    path = rocm_device_memory._vram_used_file(0)
    assert path is not None
    with open(path) as f:
        return int(f.read())


def _exact_free() -> int:
    free = rocm_device_memory.exact_free_memory(DEVICE)
    assert free is not None
    return free


def _hidden() -> int:
    """Free memory the exact count shows and HIP's free memory does not."""
    return _exact_free() - torch.accelerator.get_memory_info(DEVICE)[0]


@pytest.fixture
def exact_count():
    torch.zeros(1, device=DEVICE)
    if rocm_device_memory._vram_used_file(0) is None:
        pytest.skip("no exact VRAM count for this GPU")
    # Wait until earlier frees are erased and the count holds.
    torch.accelerator.empty_cache()
    torch.accelerator.synchronize()
    start = held_since = time.monotonic()
    last = _used()
    while time.monotonic() - held_since < 0.1:
        assert time.monotonic() - start < 10, "device memory never settled"
        time.sleep(0.005)
        if (used := _used()) != last:
            held_since, last = time.monotonic(), used


def test_exact_count_grows_with_an_allocation(exact_count):
    before = _used()
    tensor = torch.empty(64 * MB, dtype=torch.uint8, device=DEVICE)
    torch.accelerator.synchronize()
    grown = _used() - before
    del tensor
    # 64 MB, give or take driver memory (page tables).
    assert abs(grown - 64 * MB) < 32 * MB


@pytest.fixture
def code_objects():
    """Loads n copies of a small kernel code object; free memory counts each
    as at least 2 MB."""
    import triton
    import triton.language as tl

    @triton.jit
    def add_one(x_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(x_ptr + i, tl.load(x_ptr + i, mask=i < n) + 1, mask=i < n)

    x = torch.zeros(128, device=DEVICE)
    image = add_one.warmup(x, 128, BLOCK=128, grid=(1,)).asm["hsaco"]
    with open("/proc/self/maps") as f:
        path = next(line.split()[-1] for line in f if "libamdhip64" in line)
    hip = ctypes.CDLL(path)
    buffer = ctypes.create_string_buffer(image, len(image))
    modules: list[ctypes.c_void_p] = []

    def load(n: int) -> None:
        for _ in range(n):
            module = ctypes.c_void_p()
            assert hip.hipModuleLoadData(ctypes.byref(module), buffer) == 0
            modules.append(module)
        torch.accelerator.synchronize()

    yield load
    for module in modules:
        hip.hipModuleUnload(module)


def test_code_objects_are_added_back(exact_count, code_objects):
    """200 code objects of a few KB, each counted as at least 2 MB, are added
    back; another process taking 1 GB meanwhile is not."""
    hidden_before = _hidden()
    code_objects(200)
    free_alone, added_alone = _exact_free(), _hidden() - hidden_before
    assert added_alone > 200 * MB

    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, torch\n"
            "t = torch.empty(1 << 30, dtype=torch.uint8, device='cuda:0')\n"
            "torch.accelerator.synchronize()\n"
            "print('ready', flush=True)\n"
            "sys.stdin.read()\n",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        free_shared, added_shared = _exact_free(), _hidden() - hidden_before
    finally:
        assert child.stdin is not None
        child.stdin.close()
        child.wait(timeout=60)
    assert free_alone - free_shared >= 1 << 30
    # The child's own rounding (its runtime's code objects) is added back too,
    # but not its 1 GB.
    assert -32 * MB < added_shared - added_alone < 256 * MB


def _corrected_target(code_objects) -> int:
    """Corrected free memory after loading code objects, less 256 MB."""
    hidden_before = _hidden()
    code_objects(200)
    assert _hidden() - hidden_before > 200 * MB
    return _exact_free() - 256 * MB


@pytest.mark.optional
def test_corrected_free_memory_can_be_allocated(exact_count, code_objects):
    """All but 256 MB of the corrected free memory can be allocated through
    torch in 64 MB and 2 MB blocks. Fills nearly the whole GPU: runs with
    --optional."""
    target = _corrected_target(code_objects)
    chunks, allocated = [], 0
    try:
        for step in (64 * MB, 2 * MB):
            while allocated + step <= target:
                chunks.append(torch.empty(step, dtype=torch.uint8, device=DEVICE))
                allocated += step
        torch.accelerator.synchronize()
    finally:
        del chunks
        torch.accelerator.empty_cache()
    assert allocated > target - 2 * MB


@pytest.mark.optional
def test_corrected_free_memory_fits_one_allocation(exact_count, code_objects):
    """All but 256 MB of the corrected free memory can be committed as one
    driver allocation, as a defragmented extensible KV cache segment is.
    Fills nearly the whole GPU: runs with --optional."""
    target = _corrected_target(code_objects) // (2 * MB) * (2 * MB)
    tensor = ExtensibleTensor(target, device=DEVICE)
    try:
        tensor.resize_per_segment_(target)
        assert tensor.num_physical_chunks == 1
        tensor.full_view()[-1:].zero_()
        torch.accelerator.synchronize()
    finally:
        tensor.free()
