# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.v1.worker.cpu.shm  # noqa: F401
from vllm.v1.worker.cpu.buffer_utils import FusedStagedWriter, StagedWriteTensor


def test_cpu_staged_write_tensor_applies_ragged_writes():
    tensor = StagedWriteTensor(
        (4, 8),
        dtype=torch.int32,
        device=torch.device("cpu"),
    )

    tensor.stage_write(2, 3, [3, 1, 2])
    tensor.stage_write(0, 1, [-1, -2, -5])
    tensor.stage_write_elem(1, 7)
    tensor.apply_write()

    assert tensor.gpu.tolist() == [
        [0, -1, -2, -5, 0, 0, 0, 0],
        [7, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 1, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert tensor._staged_write_indices == []
    assert tensor._staged_write_starts == []
    assert tensor._staged_write_contents == []
    assert tensor._staged_write_cu_lens == []


def test_cpu_staged_write_tensor_applies_float_writes():
    tensor = StagedWriteTensor(
        (2, 4),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    tensor.stage_write(1, 1, [0.5, -1.25])
    tensor.apply_write()

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, -1.25, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(tensor.gpu, expected)


def test_cpu_fused_staged_writer_applies_each_tensor():
    first = StagedWriteTensor(
        (2, 4),
        dtype=torch.int32,
        device=torch.device("cpu"),
    )
    second = StagedWriteTensor(
        (2, 4),
        dtype=torch.int32,
        device=torch.device("cpu"),
    )
    first.stage_write(0, 0, [10, 11])
    second.stage_write(1, 2, [20, 21])

    writer = FusedStagedWriter(torch.device("cpu"), max_writes=4)
    writer.apply(
        [first, second],
        output_ptrs=torch.empty(2, dtype=torch.uint64),
        output_strides=torch.empty(2, dtype=torch.int64),
    )

    assert first.gpu.tolist() == [
        [10, 11, 0, 0],
        [0, 0, 0, 0],
    ]
    assert second.gpu.tolist() == [
        [0, 0, 0, 0],
        [0, 0, 20, 21],
    ]
    assert first._staged_write_indices == []
    assert second._staged_write_indices == []
