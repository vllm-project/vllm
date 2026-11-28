# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for direct CUDA driver calls and handle-launched Triton kernels."""

import ctypes
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops import cuda_direct
from vllm.v1.attention.ops.cuda_direct import (
    POINTER_ALIGNMENT,
    CopyList,
    CudaDriver,
    DirectKernel,
    Int32Table,
    KernelLaunch,
    direct_launch,
)

cuda = pytest.mark.skipif(
    not (current_platform.is_cuda() and torch.cuda.is_available()),
    reason="needs a CUDA GPU",
)

COPIES = [(0x100, 0x900, 32), (0x300, 0xB00, 8), (0x400, 0xC00, 16)]


def test_copy_list_drops_empty_copies():
    copies = CopyList([(0x100, 0x900, 32), (0x200, 0xA00, 0), (0x300, 0xB00, 8)])
    assert (copies.count, copies.nbytes) == (2, 40)
    assert list(copies.destinations) == [0x100, 0x300]
    assert list(copies.sources) == [0x900, 0xB00]
    assert list(copies.sizes) == [32, 8]
    assert CopyList([]).count == 0 and CopyList([(1, 2, 0)]).copies == []


def _fake_driver(batch_result: int | None) -> tuple[CudaDriver, list]:
    driver = CudaDriver.__new__(CudaDriver)
    calls: list = []

    def memcpy(dst, src, size, stream) -> int:
        calls.append(("one", dst, src, size, stream))
        return 0

    def memcpy_batch(dsts, srcs, sizes, count, *rest) -> int:
        calls.append(
            ("batch", list(zip(dsts[:count], srcs[:count], sizes[:count])), rest[-1])
        )
        return batch_result or 0

    driver._memcpy = memcpy
    driver._memcpy_batch = None if batch_result is None else memcpy_batch
    driver._attributes_ref = driver._attribute_index = None
    return driver, calls


def test_memcpy_list_batches_and_falls_back():
    driver, calls = _fake_driver(0)
    driver.memcpy_list(CopyList(COPIES), 7)
    assert calls == [("batch", COPIES, 7)]
    calls.clear()
    driver.memcpy_list(CopyList(COPIES[:1]), 7)
    driver.memcpy_list(CopyList([]), 7)
    assert calls == [("one", *COPIES[0], 7)]

    for result in (None, 801):
        driver, calls = _fake_driver(result)
        driver.memcpy_list(CopyList(COPIES), 7)
        singles = [("one", *copy, 7) for copy in COPIES]
        assert [c for c in calls if c[0] == "one"] == singles
        calls.clear()
        driver.memcpy_list(CopyList(COPIES), 7)
        assert calls == singles


def test_int32_table_writes_in_place_and_before_slots():
    driver = CudaDriver.__new__(CudaDriver)
    seen: list = []

    def batch_mem_op(stream, count, params, flags) -> int:
        ops = ctypes.cast(
            params, ctypes.POINTER(cuda_direct._MemOpParams * count)
        ).contents
        seen.append(
            (
                stream,
                [
                    (
                        op.write_value.operation,
                        op.write_value.address,
                        op.write_value.value,
                    )
                    for op in ops
                ],
            )
        )
        return 0

    driver._batch_mem_op = batch_mem_op
    tensor = torch.zeros(9, dtype=torch.int32)
    base, write = tensor.data_ptr(), CudaDriver.OP_WRITE_VALUE_32

    table = Int32Table(driver, tensor)
    table.write(5, 0, [11, 12, 13])
    assert seen == [(5, [(write, base + 4 * i, 11 + i) for i in range(3)])]
    seen.clear()
    values = [7, 0, 2**31 - 1, 9, 1]
    table.write(6, 4, values)
    assert seen == [(6, [(write, base + 4 * (4 + i), v) for i, v in enumerate(values)])]

    seen.clear()
    outside = Int32Table(driver, tensor, before=(0x7000, 0x7004))
    outside.write(8, -2, [2**32 - 1, 5, 21, 22])
    assert seen == [
        (
            8,
            [
                (write, 0x7000, 2**32 - 1),
                (write, 0x7004, 5),
                (write, base, 21),
                (write, base + 4, 22),
            ],
        )
    ]
    seen.clear()
    outside.write(8, 3, [4])
    assert seen == [(8, [(write, base + 12, 4)])]


def _fake_kernel(names: list[str], aligned: set[str]) -> SimpleNamespace:
    launches: list = []

    def launch(fn, gx, gy, gz, bx, by, bz, shared, stream, args, extra):
        slots = [
            ctypes.c_uint64.from_address(args[i]).value for i in range(len(names) + 2)
        ]
        launches.append((fn, (gx, gy, gz), (bx, by, bz), shared, stream, slots))
        return 0

    return SimpleNamespace(
        name="k",
        names=tuple(names),
        index={n: i for i, n in enumerate(names)},
        masks=[POINTER_ALIGNMENT - 1 if n in aligned else 0 for n in names] + [0, 0],
        driver=SimpleNamespace(_launch=launch, check=CudaDriver.check),
        function=0xF00,
        threads=128,
        shared=4096,
        launches=launches,
    )


def test_kernel_launch_parameters_alignment_and_isolation():
    kernel = _fake_kernel(["data_ptr", "table_ptr", "rows", "start"], {"data_ptr"})
    launch = KernelLaunch(kernel)  # type: ignore[arg-type]
    launch.update(data_ptr=0x7F00_0000_1000, table_ptr=0x7F00_0000_2004, rows=5)
    launch.set("start", 2**33 + 1)
    launch(9, 3, 2)
    assert kernel.launches[0] == (
        0xF00,
        (3, 2, 1),
        (128, 1, 1),
        4096,
        9,
        [0x7F00_0000_1000, 0x7F00_0000_2004, 5, 2**33 + 1, 0, 0],
    )
    launch.set("rows", -1)
    assert ctypes.c_int32.from_address(ctypes.addressof(launch.slots) + 16).value == -1
    with pytest.raises(ValueError, match="data_ptr"):
        launch.set("data_ptr", 0x1004)

    k2 = _fake_kernel(["data_ptr", "rows"], set())
    a, b = KernelLaunch(k2), KernelLaunch(k2)  # type: ignore[arg-type]
    a.set("rows", 3)
    b.set("rows", 4)
    a(1, 1)
    b(1, 1)
    assert [entry[5][1] for entry in k2.launches] == [3, 4]


def test_direct_kernel_loading_and_fallback(monkeypatch):
    class NotAKernel:
        __name__ = "not_a_kernel"

    assert not DirectKernel(NotAKernel(), {"x_ptr": torch.int32}).direct
    assert direct_launch(NotAKernel(), {"x_ptr": torch.int32}) is None

    def make_jit(shared: int = 0):
        def param(name, integer):
            return SimpleNamespace(
                name=name,
                is_constexpr=False,
                annotation_type="i32" if integer else "",
                do_not_specialize=integer,
                do_not_specialize_on_alignment=False,
            )

        compiled = SimpleNamespace(
            run=None,
            function=0xF00,
            metadata=SimpleNamespace(num_warps=4, shared=shared),
            src=SimpleNamespace(signature={"x_ptr": "*i32", "n": "i32"}),
        )
        return SimpleNamespace(
            __name__="k",
            params=[param("x_ptr", False), param("n", True)],
            warmup=lambda *a, grid, **kw: compiled,
        )

    driver = CudaDriver.__new__(CudaDriver)
    monkeypatch.setattr(cuda_direct, "cuda_driver", lambda: driver)
    driver._param_info = lambda fn, idx, off, sz: int(idx >= 4)
    k = DirectKernel(make_jit(), {"x_ptr": torch.int32})
    assert (
        k.direct
        and k.names == ("x_ptr", "n")
        and (k.function, k.threads) == (0xF00, 128)
    )
    assert not DirectKernel(make_jit(shared=229 << 10), {"x_ptr": torch.int32}).direct

    class OlderDriver:
        def __getattr__(self, name: str):
            if name == "cuFuncGetParamInfo":
                raise AttributeError(name)
            return SimpleNamespace()

    monkeypatch.setattr(ctypes, "CDLL", lambda name: OlderDriver())
    with pytest.raises(RuntimeError, match="kernel parameters"):
        CudaDriver().param_count(0xF00)
    monkeypatch.setattr(cuda_direct, "cuda_driver", CudaDriver)
    assert not DirectKernel(make_jit(), {"x_ptr": torch.int32}).direct


@cuda
def test_direct_launch_matches_triton_launch():
    from tests.v1.attention import kv_ce_triton_launch as triton_ref
    from vllm.triton_utils import triton
    from vllm.v1.attention.ops import kv_ce_rows

    device, g = "cuda", torch.Generator().manual_seed(0)
    new, pad, start, dcp, interleave, per_rank = 3000, 24, 130_001, 4, 16, 1024
    restore = torch.randperm(new + pad, generator=g)[:new].to(device)
    want = torch.full((new + pad,), -1, dtype=torch.int64, device=device)
    triton_ref.stage_slots(restore, want, start, interleave, dcp, per_rank)
    k_slots = kv_ce_rows.stage_slots_kernel()
    assert k_slots is not None and k_slots.direct
    slots = torch.empty_like(want)
    stream = torch.cuda.current_stream().cuda_stream
    k_slots.driver.fill(slots.data_ptr(), 0xFF, 8 * slots.numel(), stream)
    k_slots.bind(
        restore_ptr=restore.data_ptr(),
        slots_ptr=slots.data_ptr(),
        context_start=start,
        new_rows=new,
        interleave=interleave,
        dcp_size=dcp,
        per_rank=per_rank,
    )(stream, triton.cdiv(new, kv_ce_rows.SLOT_BLOCK))
    torch.accelerator.synchronize()
    assert torch.equal(slots, want)

    entry, row = kv_ce_rows.FP8_ENTRY, kv_ce_rows.ROW_DIM
    cache = torch.randint(0, 256, (8, 16, entry), generator=g, dtype=torch.uint8).to(
        device
    )
    cache.view(-1, entry)[:, 512:528] = (
        torch.ones(128, 4).view(torch.uint8).view(128, 16).to(device)
    )
    i32 = {"dtype": torch.int32, "device": device}
    sources = torch.tensor([0, 40, 64, 100], **i32)
    targets = torch.tensor([3, 70, 120, 180], **i32)
    counts = torch.tensor([33, 0, 16, 20], **i32)
    expected = torch.zeros((220, row), dtype=torch.bfloat16, device=device)
    triton_ref.stage_rows(cache, sources, targets, counts, expected, 33)
    k_rows = kv_ce_rows.stage_rows_kernel()
    assert k_rows is not None
    region = torch.zeros_like(expected)
    k_rows.bind(
        cache_ptr=cache.data_ptr(),
        sources_ptr=sources.data_ptr(),
        targets_ptr=targets.data_ptr(),
        counts_ptr=counts.data_ptr(),
        region_ptr=region.data_ptr(),
    )(stream, triton.cdiv(33, kv_ce_rows.STAGE_ROWS), 4)
    torch.accelerator.synchronize()
    assert torch.equal(region.view(torch.int16), expected.view(torch.int16))
    assert int(expected.view(torch.int16).ne(0).sum()) > 0
