# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.hy_v4_flash.nvidia.indexpool_mqa import score_indexpool_groups
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(89),
    reason="HY4 Flash IndexPool Triton fallback is enabled on SM89",
)


def test_paged_fp8_scoring_on_ada():
    """Check nontrivial page mappings, strided pages and ragged bounds."""
    torch.manual_seed(29)
    page, dim, heads, width = 64, 128, 32, 128
    q = torch.randn((2, 1, heads, dim), device="cuda").to(torch.float8_e4m3fn)
    keys = torch.randn((4, page, dim), device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randn((2, heads), device="cuda")
    weights[:, 16:] = 0  # HY4 Flash's 16 real heads in 32 padded slots.
    scales = (2.0 ** torch.arange(-3, 5, device="cuda")).repeat(4, 8)
    pages = torch.empty((4, page * (dim + 4) + 32), device="cuda", dtype=torch.uint8)
    cache = pages[:, :-32]
    cache[:, : page * dim] = keys.view(torch.uint8).flatten(1)
    cache[:, page * dim :] = scales.view(torch.uint8)
    table = torch.tensor([[3, 0], [1, 2]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([[66], [127]], dtype=torch.int32, device="cuda")
    out = score_indexpool_groups(
        q,
        None,
        cache.view(4, page, dim + 4),
        weights,
        lengths,
        table,
        width,
        0,
        None,
        use_triton=True,
    )
    for r, n in enumerate((66, 127)):
        gathered = keys.float()[table[r].long()].reshape(width, dim)[:n]
        # CPU reference avoids using TF32 and shares no scoring code.
        logits = q[r, 0].float().cpu() @ gathered.float().cpu().T
        scale = scales[table[r].long()].flatten()[:n].cpu()
        expected = (logits.relu() * weights[r].cpu()[:, None]).sum(dim=0) * scale
        torch.testing.assert_close(out[r, :n].cpu(), expected, rtol=2e-5, atol=5e-4)
