# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-only test.", allow_module_level=True)

from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _per_sequence_context_lens


def test_per_row_table_uses_last_column():
    seq_lens = torch.tensor([8192, 1500, 3000, 700], dtype=torch.int32)
    next_n = 3
    offsets = torch.arange(next_n, dtype=torch.int32) - next_n + 1
    per_row = seq_lens.unsqueeze(1) + offsets

    out = _per_sequence_context_lens(per_row)

    assert out.shape == (4,)
    assert out.is_contiguous()
    assert torch.equal(out, seq_lens)


@pytest.mark.parametrize("shape", [(4,), (4, 1)])
def test_per_sequence_input_is_unchanged(shape):
    context_lens = torch.arange(1, 5, dtype=torch.int32).reshape(shape)
    assert _per_sequence_context_lens(context_lens) is context_lens
