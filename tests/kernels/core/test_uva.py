# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.v1.worker.gpu import buffer_utils
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

DEVICE_TYPE = current_platform.device_type
DEVICES = [
    f"{DEVICE_TYPE}:{i}"
    for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", DEVICES)
def test_cpu_write(device):
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(10, 10, device="cpu", pin_memory=True, dtype=torch.int32)
    gpu_view = get_accelerator_view_from_cpu_tensor(cpu_tensor)
    assert gpu_view.device.type == DEVICE_TYPE

    assert gpu_view[0, 0] == 0
    assert gpu_view[2, 3] == 0
    assert gpu_view[4, 5] == 0

    cpu_tensor[0, 0] = 1
    cpu_tensor[2, 3] = 2
    cpu_tensor[4, 5] = -1

    gpu_view.mul_(2)
    assert gpu_view[0, 0] == 2
    assert gpu_view[2, 3] == 4
    assert gpu_view[4, 5] == -2


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", DEVICES)
def test_gpu_write(device):
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(10, 10, device="cpu", pin_memory=True, dtype=torch.int32)
    gpu_view = get_accelerator_view_from_cpu_tensor(cpu_tensor)
    assert gpu_view.device.type == DEVICE_TYPE

    assert gpu_view[0, 0] == 0
    assert gpu_view[2, 3] == 0
    assert gpu_view[4, 5] == 0

    gpu_view[0, 0] = 1
    gpu_view[2, 3] = 2
    gpu_view[4, 5] = -1
    gpu_view.mul_(2)

    assert cpu_tensor[0, 0] == 2
    assert cpu_tensor[2, 3] == 4
    assert cpu_tensor[4, 5] == -2


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", DEVICES)
def test_staged_write_uses_uva_contents_for_uva_target(device, monkeypatch):
    def fail_async_tensor_h2d(*args, **kwargs):
        pytest.fail("UVA-backed targets should not copy write contents to the GPU")

    monkeypatch.setattr(buffer_utils, "async_tensor_h2d", fail_async_tensor_h2d)
    staged = StagedWriteTensor(
        (3, 4096),
        dtype=torch.int32,
        device=torch.device(device),
        max_concurrency=2,
        uva_instead_of_gpu=True,
    )

    staged.stage_write(2, 3, [11, 12, 13])
    staged.apply_write()
    torch.accelerator.synchronize()
    staged.stage_write(1, 7, [21, 22])
    staged.apply_write()
    torch.accelerator.synchronize()
    staged.stage_write(0, 1020, range(1500))
    staged.apply_write()
    torch.accelerator.synchronize()

    assert staged.gpu[2, 3:6].tolist() == [11, 12, 13]
    assert staged.gpu[1, 7:9].tolist() == [21, 22]
    assert staged.gpu[0, 1020:2520].tolist() == list(range(1500))


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("input_type", ["list", "numpy", "tensor"])
@pytest.mark.parametrize("size", [(4,), (4, 3)])
def test_uva_pool_overwrites_exposed_prefix(input_type, size):
    """Both slots expose only current contents across growth and shorter reuse."""
    pool = buffer_utils.UvaBufferPool(size, torch.int32, max_concurrency=2)
    for buf in pool._uva_bufs:
        assert tuple(buf.cpu.shape) == size
        assert buf.cpu.is_pinned()
        assert torch.count_nonzero(buf.cpu).item() == 0
    lengths = [0, 0, 3, 3, 4, 4, 5, 3, 3, 5, 1024, 1024, 1025, 1025, 2, 2]
    for step, length in enumerate(lengths):
        if input_type == "list" and len(size) > 1 and length == 0:
            # An empty list has no trailing shape; preserve NumPy's rejection.
            with pytest.raises(ValueError, match="could not broadcast"):
                pool.copy_to_uva([])
            continue
        shape = (length, *size[1:])
        expected = (
            torch.arange(int(np.prod(shape)), dtype=torch.int32, device="cpu").reshape(
                shape
            )
            - step
        )
        values = expected.tolist()
        if input_type == "numpy":
            values = expected.numpy()
        elif input_type == "tensor":
            values = expected
        before = list(pool._uva_bufs)
        slot = (pool._curr + 1) % pool.max_concurrency
        result = pool.copy_to_uva(values)
        assert tuple(result.shape) == shape
        assert pool._curr == slot
        assert pool._uva_bufs[1 - slot] is before[1 - slot]
        if length <= before[slot].cpu.shape[0]:
            assert pool._uva_bufs[slot] is before[slot]
        else:
            assert tuple(pool._uva_bufs[slot].cpu.shape) == (
                1 << (length - 1).bit_length(),
                *size[1:],
            )
        assert pool.size == size
        # Blocking read also retires GPU readers before the next slot reuse.
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize("input_type", ["numpy", "tensor"])
def test_uva_pool_copy_to_gpu_preserves_shape_and_out(use_out, input_type):
    pool = buffer_utils.UvaBufferPool((2, 3), torch.int32, max_concurrency=2)
    for length in (2, 5, 3, 6):
        expected = torch.arange(length * 3, dtype=torch.int32, device="cpu").reshape(
            length, 3
        )
        values = expected.numpy() if input_type == "numpy" else expected
        out = (
            torch.empty(expected.shape, dtype=torch.int32, device=DEVICE_TYPE)
            if use_out
            else None
        )
        result = pool.copy_to_gpu(values, out=out)
        if use_out:
            assert result is out
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("uva_target", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_staged_write_inflight(uva_target, dtype):
    """Preserve every generation until its consumer finishes before slot reuse."""
    device = torch.device(f"{DEVICE_TYPE}:0")
    with torch.accelerator.device_index(device.index):
        state = StagedWriteTensor(
            (4, 4096),
            dtype,
            device,
            max_concurrency=2,
            uva_instead_of_gpu=uva_target,
        )
        assert (state.write_contents is not None) == uva_target
        stream = torch.Stream(device=device)
        stream.wait_stream(torch.accelerator.current_stream())
        pending: list[tuple[torch.Event, torch.Tensor, torch.Tensor]] = []
        expected = torch.zeros((4, 4096), dtype=dtype, device="cpu")
        for step in range(24):
            if len(pending) == 2:
                event, snapshot, reference = pending.pop(0)
                event.synchronize()
                torch.testing.assert_close(snapshot.cpu(), reference, rtol=0, atol=0)
            # Growing and shrinking lengths exercise reallocation and reuse.
            length = [3, 17, 1500, 4090][step % 4]
            row = step % 3
            values = torch.arange(length, dtype=dtype, device="cpu") + step * 8192
            if dtype == torch.float32:
                values += 0.25
            expected[row, 2 : 2 + length] = values
            expected[3, 1:4] = step
            with stream:
                state.stage_write(row, 2, values.tolist())
                state.stage_write(3, 1, [step] * 3)
                state.apply_write()
                # A GPU consumer observes this generation before the next update.
                snapshot = state.gpu.clone()
                event = torch.Event()
                event.record(stream)
            pending.append((event, snapshot, expected.clone()))
        for event, snapshot, reference in pending:
            event.synchronize()
            torch.testing.assert_close(snapshot.cpu(), reference, rtol=0, atol=0)


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", DEVICES)
def test_non_pinned_cpu_tensor(device):
    # Non-pinned CPU tensors are internally copied into a pinned buffer,
    # so the resulting gpu view reflects the values at creation time but
    # is decoupled from further writes to the original `cpu_tensor`.
    torch.set_default_device(device)
    cpu_tensor = torch.arange(100, dtype=torch.int32, device="cpu").view(10, 10)
    assert not cpu_tensor.is_pinned()
    gpu_view = get_accelerator_view_from_cpu_tensor(cpu_tensor)
    assert gpu_view.device.type == DEVICE_TYPE

    assert gpu_view[0, 0] == 0
    assert gpu_view[2, 3] == 23
    assert gpu_view[9, 9] == 99

    # Writes to the original (unpinned) CPU tensor must not affect the view,
    # since a private pinned copy was made.
    cpu_tensor[0, 0] = -1
    assert gpu_view[0, 0] == 0

    # The view itself remains writable and independently usable.
    gpu_view.mul_(2)
    assert gpu_view[2, 3] == 46
    assert gpu_view[9, 9] == 198


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.skipif(
    not current_platform.is_xpu(), reason="XPU non-contiguous UVA test."
)
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_non_contiguous_strided_view(device, pinned):
    torch.set_default_device(device)
    # Simulate scale_kn: a [32, 16] contiguous tensor
    scale_kn = torch.arange(
        512, dtype=torch.float32, device="cpu", pin_memory=pinned
    ).view(32, 16)
    # Transposed view: shape [16, 32], non-contiguous stride (1, 16)
    cpu_view = scale_kn.t()
    assert cpu_view.shape == (16, 32)
    assert cpu_view.stride() == (1, 16)
    assert not cpu_view.is_contiguous()

    gpu_view = get_accelerator_view_from_cpu_tensor(cpu_view)
    assert gpu_view.device.type == DEVICE_TYPE
    assert gpu_view.shape == (16, 32)
    assert gpu_view.stride() == (1, 16)
    assert not gpu_view.is_contiguous()

    # Transposing tensor_view back should yield a contiguous [32, 16] tensor
    gpu_view_t = gpu_view.t()
    assert gpu_view_t.shape == (32, 16)
    assert gpu_view_t.is_contiguous()

    # Correctness check: values match original transposed CPU tensor
    assert torch.equal(gpu_view.cpu(), cpu_view)
