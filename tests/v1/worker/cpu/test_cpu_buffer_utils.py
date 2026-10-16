# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.cpu.buffer_utils import (
    UvaBuffer,
    _fused_staged_writer_apply,
    _staged_write_tensor_apply_write,
)
from vllm.v1.worker.gpu import buffer_utils as buffer_utils_module
from vllm.v1.worker.gpu.buffer_utils import FusedStagedWriter, StagedWriteTensor


@pytest.fixture
def cpu_staged_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(buffer_utils_module, "UvaBuffer", UvaBuffer)
    monkeypatch.setattr(
        buffer_utils_module.StagedWriteTensor,
        "apply_write",
        _staged_write_tensor_apply_write,
    )
    monkeypatch.setattr(
        buffer_utils_module.FusedStagedWriter,
        "apply",
        _fused_staged_writer_apply,
    )


@pytest.mark.parametrize(
    ("size", "writes", "expected"),
    [
        (
            4,
            [(2, 0, [7]), (0, 0, [9])],
            torch.tensor([9, 0, 7, 0], dtype=torch.int32),
        ),
        (
            (3, 8),
            [
                (2, 1, [10, 11, 12]),
                (0, 3, [20, 21]),
                (2, 5, [30, 31]),
            ],
            torch.tensor(
                [
                    [0, 0, 0, 20, 21, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 10, 11, 12, 0, 30, 31, 0],
                ],
                dtype=torch.int32,
            ),
        ),
    ],
)
def test_staged_write_tensor_apply_write(
    cpu_staged_writes: None,
    size: int | tuple[int, int],
    writes: list[tuple[int, int, list[int]]],
    expected: torch.Tensor,
) -> None:
    staged = StagedWriteTensor(size, torch.int32, torch.device("cpu"))
    for row_idx, start_idx, contents in writes:
        staged.stage_write(row_idx, start_idx, contents)

    staged.apply_write()

    assert torch.equal(staged.gpu, expected)
    assert not staged._staged_write_indices
    assert not staged._staged_write_starts
    assert not staged._staged_write_contents
    assert not staged._staged_write_cu_lens


def test_fused_staged_writer_apply(cpu_staged_writes: None) -> None:
    first = StagedWriteTensor((2, 4), torch.int32, torch.device("cpu"))
    second = StagedWriteTensor((2, 4), torch.int32, torch.device("cpu"))
    first.stage_write(1, 1, [4, 5])
    second.stage_write(0, 2, [8, 9])

    writer = FusedStagedWriter(torch.device("cpu"), max_writes=2)
    output_ptrs = torch.tensor(
        [first.gpu.data_ptr(), second.gpu.data_ptr()], dtype=torch.uint64
    )
    output_strides = torch.tensor(
        [first.gpu.stride(0), second.gpu.stride(0)], dtype=torch.int64
    )
    writer.apply([first, second], output_ptrs, output_strides)

    assert first.gpu.tolist() == [[0, 0, 0, 0], [0, 4, 5, 0]]
    assert second.gpu.tolist() == [[0, 0, 8, 9], [0, 0, 0, 0]]
