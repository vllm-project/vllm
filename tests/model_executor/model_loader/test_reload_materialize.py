# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialized reload temporaries start zeroed: rows no checkpoint tensor covers
(alignment padding a module zeroed at init) must not carry stale memory."""

import pytest
import torch

from vllm.model_executor.model_loader.reload.meta import (
    materialize_meta_tensor,
    to_meta_tensor,
)


class TestMaterializeMetaTensor:
    @pytest.mark.parametrize(
        "shape,transpose,dtype",
        [
            ((8, 16), False, torch.bfloat16),
            ((8, 16), True, torch.float32),
            ((4, 6, 10), False, torch.uint8),
        ],
        ids=["bf16", "strided-fp32", "packed-uint8"],
    )
    def test_zero_filled(self, shape, transpose, dtype):
        param = torch.nn.Parameter(torch.ones(shape, dtype=dtype), requires_grad=False)
        if transpose:
            param = torch.nn.Parameter(param.data.t(), requires_grad=False)
        param.marker = "kept"

        out = materialize_meta_tensor(to_meta_tensor(param))

        assert not out.is_meta
        assert out.stride() == param.stride()
        assert out.marker == "kept"
        assert type(out) is type(param)
        assert torch.count_nonzero(out) == 0
