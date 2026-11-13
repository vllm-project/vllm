# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M-RoPE positions follow global token indices in each local batch."""

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.mm.rope import RopeState


@pytest.mark.parametrize(
    "idx_mapping,query_start_loc,input_positions,prefill_lens,expected",
    [
        pytest.param(
            [0],
            [0, 3],
            [2, 3, 4],
            [8, 4],
            [[112, 113, 114], [122, 123, 124], [132, 133, 134]],
            id="chunked-prefill",
        ),
        pytest.param(
            [0, 0],
            [0, 2, 4],
            [0, 1, 3, 4],
            [8, 4],
            [[110, 111, 113, 114], [120, 121, 123, 124], [130, 131, 133, 134]],
            id="pcp-fragments",
        ),
        pytest.param(
            [0],
            [0, 2],
            [1, 4],
            [8, 4],
            [[111, 114], [121, 124], [131, 134]],
            id="noncontiguous-prefill",
        ),
        pytest.param(
            [1],
            [0, 2],
            [4, 5],
            [8, 4],
            [[6, 7]] * 3,
            id="decode-delta",
        ),
        pytest.param(
            [1, 1],
            [0, 1, 2],
            [4, 6],
            [8, 4],
            [[6, 8]] * 3,
            id="decode-fragments",
        ),
        pytest.param(
            [1, 0],
            [0, 2, 4],
            [4, 5, 2, 3],
            [8, 4],
            [[6, 7, 112, 113], [6, 7, 122, 123], [6, 7, 132, 133]],
            id="mixed-reordered-requests",
        ),
        pytest.param(
            [0, 1],
            [0, 0, 2],
            [4, 5],
            [8, 4],
            [[6, 7]] * 3,
            id="empty-row",
        ),
    ],
)
def test_prepare_positions(
    idx_mapping, query_start_loc, input_positions, prefill_lens, expected
):
    device = torch.device(current_platform.device_type)
    rope = RopeState(3, 2, 8, 8, device)
    for req_idx in range(2):
        for dim in range(3):
            base = 100 * (req_idx + 1) + 10 * (dim + 1)
            rope.prefill_positions.stage_write(
                req_idx * 3 + dim, 0, np.arange(base, base + 8, dtype=np.int32)
            )
    rope.prefill_delta.np[:] = [0, 2]
    rope.apply_staged_writes()

    def tensor(values, dtype=torch.int64):
        return torch.tensor(values, dtype=dtype, device=device)

    rope.prepare_positions(
        tensor(idx_mapping),
        tensor(query_start_loc, torch.int32),
        tensor(prefill_lens, torch.int32),
        tensor(input_positions),
    )
    result = rope.get_positions(len(input_positions))
    assert torch.equal(result, tensor(expected))
