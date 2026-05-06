# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm._custom_ops import permute_cols

if not hasattr(torch.ops._C, "permute_cols"):
    pytest.skip(reason="permute_cols is not supported on ROCm", allow_module_level=True)


@pytest.mark.parametrize("shape", [(1, 512), (544, 4096), (67, 8192)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_permute_cols(shape, dtype):
    x = torch.randn(shape, dtype=dtype).cuda()
    perm = torch.randperm(x.shape[1]).to(torch.int).cuda()
    opcheck(torch.ops._C.permute_cols, (x, perm))
    y = permute_cols(x, perm)
    torch.testing.assert_close(y, x[:, perm])
