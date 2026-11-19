# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The driver calls and direct kernel launches of the PCP + DCP overlaps
(`cuda_direct`): an argument array or a parameter block that is built once and
then only edited must hold exactly what the per-call path would pass."""

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

# CUDA only: direct launches go through the driver library (libcuda.so.1).
cuda = pytest.mark.skipif(
    not (current_platform.is_cuda() and torch.cuda.is_available()),
    reason="needs a CUDA GPU",
)


def test_a_copy_list_keeps_the_copies_in_order_without_the_empty_ones():
    copies = CopyList([(0x100, 0x900, 32), (0x200, 0xA00, 0), (0x300, 0xB00, 8)])
    assert copies.count == 2 and copies.nbytes == 40
    assert list(copies.destinations) == [0x100, 0x300]
    assert list(copies.sources) == [0x900, 0xB00]
    assert list(copies.sizes) == [32, 8]
    assert CopyList([]).count == 0 and CopyList([(1, 2, 0)]).copies == []


def _driver(batch_result: int | None) -> tuple[CudaDriver, list]:
    """A driver whose copy calls are recorded; `batch_result` is what the batched
    copy returns, None where the driver has none."""
    driver = CudaDriver.__new__(CudaDriver)
    calls: list = []

    def memcpy(destination, source, size, stream) -> int:
        calls.append(("one", destination, source, size, stream))
        return 0

    def memcpy_batch(destinations, sources, sizes, count, *rest) -> int:
        triples = list(zip(destinations[:count], sources[:count], sizes[:count]))
        calls.append(("batch", triples, rest[-1]))
        return batch_result or 0

    driver._memcpy = memcpy
    driver._memcpy_batch = None if batch_result is None else memcpy_batch
    driver._attributes_ref = driver._attribute_index = None
    return driver, calls


COPIES = [(0x100, 0x900, 32), (0x300, 0xB00, 8), (0x400, 0xC00, 16)]


def test_several_copies_are_one_driver_call():
    driver, calls = _driver(0)
    driver.memcpy_list(CopyList(COPIES), 7)
    assert calls == [("batch", COPIES, 7)]
    del calls[:]
    driver.memcpy_list(CopyList(COPIES[:1]), 7)
    driver.memcpy_list(CopyList([]), 7)
    assert calls == [("one", *COPIES[0], 7)]


def test_a_driver_without_the_batched_copy_copies_one_by_one():
    for result in (None, 801):
        driver, calls = _driver(result)
        driver.memcpy_list(CopyList(COPIES), 7)
        singles = [("one", *copy, 7) for copy in COPIES]
        assert [call for call in calls if call[0] == "one"] == singles
        # A rejected batch enqueued nothing and is not tried again.
        del calls[:]
        driver.memcpy_list(CopyList(COPIES), 7)
        assert calls == singles


def test_a_table_write_fills_in_the_values_of_operations_built_once():
    driver = CudaDriver.__new__(CudaDriver)
    seen: list = []

    def batch_mem_op(stream, count, params, flags) -> int:
        operations = ctypes.cast(
            params, ctypes.POINTER(cuda_direct._MemOpParams * count)
        ).contents
        writes = [op.write_value for op in operations]
        seen.append((stream, [(w.operation, w.address, w.value) for w in writes]))
        return 0

    driver._batch_mem_op = batch_mem_op
    tensor = torch.zeros(9, dtype=torch.int32)
    table = Int32Table(driver, tensor)
    base, write = tensor.data_ptr(), CudaDriver.OP_WRITE_VALUE_32
    table.write(5, 0, [11, 12, 13])
    assert seen == [(5, [(write, base + 4 * i, 11 + i) for i in range(3)])]
    del seen[:]
    table.write(6, 4, [7, 0, 2**31 - 1, 9, 1])
    values = [7, 0, 2**31 - 1, 9, 1]
    assert seen == [
        (6, [(write, base + 4 * (4 + i), value) for i, value in enumerate(values)])
    ]
    # Words outside the tensor are its elements before the first: one batch
    # carries them and the tensor's head.
    del seen[:]
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
    del seen[:]
    outside.write(8, 3, [4])
    assert seen == [(8, [(write, base + 12, 4)])]


def _kernel(names: list[str], aligned: set[str]) -> SimpleNamespace:
    """What a `KernelLaunch` needs of a loaded kernel."""
    launches: list = []

    def launch(function, gx, gy, gz, bx, by, bz, shared, stream, arguments, extra):
        count = len(names) + 2
        slots = [
            ctypes.c_uint64.from_address(arguments[position]).value
            for position in range(count)
        ]
        launches.append((function, (gx, gy, gz), (bx, by, bz), shared, stream, slots))
        return 0

    driver = SimpleNamespace(_launch=launch, check=CudaDriver.check)
    kernel = SimpleNamespace(
        name="k",
        names=tuple(names),
        index={name: position for position, name in enumerate(names)},
        masks=[POINTER_ALIGNMENT - 1 if n in aligned else 0 for n in names] + [0, 0],
        driver=driver,
        function=0xF00,
        threads=128,
        shared=4096,
    )
    kernel.launches = launches
    return kernel


def test_a_launch_passes_the_parameter_block_as_it_was_last_set():
    kernel = _kernel(["data_ptr", "table_ptr", "rows", "start"], {"data_ptr"})
    launch = KernelLaunch(kernel)  # type: ignore[arg-type]
    launch.update(data_ptr=0x7F00_0000_1000, table_ptr=0x7F00_0000_2004, rows=5)
    launch.set("start", 2**33 + 1)
    launch(9, 3, 2)
    function, grid, block, shared, stream, slots = kernel.launches[0]
    assert (function, grid, block, shared, stream) == (
        0xF00,
        (3, 2, 1),
        (128, 1, 1),
        4096,
        9,
    )
    # The two trailing pointers Triton appends stay null.
    assert slots == [0x7F00_0000_1000, 0x7F00_0000_2004, 5, 2**33 + 1, 0, 0]
    launch.set("rows", 6)
    launch(9, 1)
    assert kernel.launches[1][1] == (1, 1, 1)
    assert kernel.launches[1][5][2] == 6
    # A 32-bit parameter is the low half of its slot, negative values included.
    launch.set("rows", -1)
    assert ctypes.c_int32.from_address(ctypes.addressof(launch.slots) + 16).value == -1


def test_a_pointer_the_kernel_assumes_aligned_must_be_aligned():
    kernel = _kernel(["data_ptr", "table_ptr"], {"data_ptr"})
    launch = KernelLaunch(kernel)  # type: ignore[arg-type]
    launch.set("table_ptr", 0x1004)
    with pytest.raises(ValueError, match="data_ptr"):
        launch.set("data_ptr", 0x1004)


def test_each_binding_has_its_own_parameters():
    kernel = _kernel(["data_ptr", "rows"], set())
    first, second = KernelLaunch(kernel), KernelLaunch(kernel)  # type: ignore[arg-type]
    first.set("rows", 3)
    second.set("rows", 4)
    first(1, 1)
    second(1, 1)
    assert [launch[5][1] for launch in kernel.launches] == [3, 4]


def test_a_kernel_that_cannot_be_loaded_keeps_the_triton_launch():
    """No device, the interpreter, another Triton: the owner launches through
    Triton instead, so nothing here may raise."""

    class NotAKernel:
        __name__ = "not_a_kernel"

    kernel = DirectKernel(NotAKernel(), {"x_ptr": torch.int32})
    assert not kernel.direct
    assert direct_launch(NotAKernel(), {"x_ptr": torch.int32}) is None


def _jit_fn(shared: int = 0) -> SimpleNamespace:
    """What `DirectKernel` reads of a Triton kernel `k(x_ptr, n: tl.int32)` and
    of its compiled form, which needs `shared` bytes of shared memory."""

    def param(name: str, integer: bool) -> SimpleNamespace:
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
        warmup=lambda *args, grid, **constexprs: compiled,
    )


def test_a_kernel_is_launched_directly_only_where_one_plain_launch_serves(
    monkeypatch,
):
    """A direct launch passes the parameters and nothing else. A driver that
    cannot count a kernel's parameters, or a kernel whose launch needs an
    attribute, keeps the Triton launch."""
    driver = CudaDriver.__new__(CudaDriver)
    monkeypatch.setattr(cuda_direct, "cuda_driver", lambda: driver)
    # The kernel's two parameters and Triton's two trailing pointers.
    driver._param_info = lambda function, index, offset, size: int(index >= 4)
    kernel = DirectKernel(_jit_fn(), {"x_ptr": torch.int32})
    assert kernel.direct and kernel.names == ("x_ptr", "n")
    assert (kernel.function, kernel.threads) == (0xF00, 128)
    assert not DirectKernel(_jit_fn(shared=229 << 10), {"x_ptr": torch.int32}).direct

    class OlderDriver:
        """A driver library without the call that describes parameters."""

        def __getattr__(self, name: str) -> SimpleNamespace:
            if name == "cuFuncGetParamInfo":
                raise AttributeError(name)
            return SimpleNamespace()

    monkeypatch.setattr(ctypes, "CDLL", lambda name: OlderDriver())
    driver = CudaDriver()
    with pytest.raises(RuntimeError, match="kernel parameters"):
        driver.param_count(0xF00)
    assert not DirectKernel(_jit_fn(), {"x_ptr": torch.int32}).direct


@cuda
def test_a_direct_launch_does_what_the_triton_launch_does():
    from tests.v1.attention import kv_ce_triton_launch as launch_through_triton
    from vllm.triton_utils import triton
    from vllm.v1.attention.ops import kv_ce_rows

    device = "cuda"
    generator = torch.Generator().manual_seed(0)
    new, pad, start, dcp, interleave, per_rank = 3000, 24, 130_001, 4, 16, 1024
    restore = torch.randperm(new + pad, generator=generator)[:new].to(device)
    want = torch.full((new + pad,), -1, dtype=torch.int64, device=device)
    launch_through_triton.stage_slots(restore, want, start, interleave, dcp, per_rank)
    kernel = kv_ce_rows.stage_slots_kernel()
    assert kernel is not None and kernel.direct
    slots = torch.empty_like(want)
    stream = torch.cuda.current_stream().cuda_stream
    kernel.driver.fill(slots.data_ptr(), 0xFF, 8 * slots.numel(), stream)
    launch = kernel.bind(
        restore_ptr=restore.data_ptr(),
        slots_ptr=slots.data_ptr(),
        context_start=start,
        new_rows=new,
        interleave=interleave,
        dcp_size=dcp,
        per_rank=per_rank,
    )
    launch(stream, triton.cdiv(new, kv_ce_rows.SLOT_BLOCK))
    torch.accelerator.synchronize()
    assert torch.equal(slots, want)

    entry, row = kv_ce_rows.FP8_ENTRY, kv_ce_rows.ROW_DIM
    cache = torch.randint(
        0, 256, (8, 16, entry), generator=generator, dtype=torch.uint8
    ).to(device)
    # Finite scales, so no upconverted value is a NaN whose bits could differ.
    cache.view(-1, entry)[:, 512:528] = (
        torch.ones(128, 4).view(torch.uint8).view(128, 16).to(device)
    )
    i32 = {"dtype": torch.int32, "device": device}
    sources = torch.tensor([0, 40, 64, 100], **i32)
    targets = torch.tensor([3, 70, 120, 180], **i32)
    counts = torch.tensor([33, 0, 16, 20], **i32)
    expected = torch.zeros((220, row), dtype=torch.bfloat16, device=device)
    launch_through_triton.stage_rows(cache, sources, targets, counts, expected, 33)
    rows_kernel = kv_ce_rows.stage_rows_kernel()
    assert rows_kernel is not None
    region = torch.zeros_like(expected)
    rows_kernel.bind(
        cache_ptr=cache.data_ptr(),
        sources_ptr=sources.data_ptr(),
        targets_ptr=targets.data_ptr(),
        counts_ptr=counts.data_ptr(),
        region_ptr=region.data_ptr(),
    )(stream, triton.cdiv(33, kv_ce_rows.STAGE_ROWS), 4)
    torch.accelerator.synchronize()
    assert torch.equal(region.view(torch.int16), expected.view(torch.int16))
    assert int(expected.view(torch.int16).ne(0).sum()) > 0
