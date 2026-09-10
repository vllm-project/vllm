# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor


def _make_staging_tensor(dtype: torch.dtype) -> StagedWriteTensor:
    tensor = StagedWriteTensor.__new__(StagedWriteTensor)
    tensor._staged_write_indices = []
    tensor._staged_write_starts = []
    tensor._staged_write_contents = []
    tensor._staged_write_numel = 0
    tensor._staged_write_cu_lens = []
    tensor._staged_write_np_dtype = torch.empty(0, dtype=dtype).numpy().dtype
    return tensor


def test_staged_write_contents_preserve_order_and_dtype():
    tensor = _make_staging_tensor(torch.int32)

    tensor.stage_write(0, 0, np.asarray([1, 2], dtype=np.int64))
    tensor.stage_write(1, 3, (value for value in [3, 4]))
    tensor.stage_write_elem(2, 5)

    contents = tensor._materialize_staged_write_contents()
    assert contents.tolist() == [1, 2, 3, 4, 5]
    assert contents.dtype == np.int32
    assert tensor._staged_write_cu_lens == [2, 4, 5]


def test_clear_staged_writes_resets_chunk_bookkeeping():
    tensor = _make_staging_tensor(torch.float32)
    tensor.stage_write(0, 0, [1.5, 2.5])

    tensor.clear_staged_writes()

    assert tensor._materialize_staged_write_contents().size == 0
    assert tensor._staged_write_numel == 0
    assert tensor._staged_write_cu_lens == []
