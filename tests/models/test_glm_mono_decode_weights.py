# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v32.amd.mono_decode import split_kv_b


def _dequant(w: torch.Tensor, s: torch.Tensor, bm: int, bk: int) -> torch.Tensor:
    rows, cols = w.shape
    scale = s.repeat_interleave(bm, 0)[:rows].repeat_interleave(bk, 1)[:, :cols]
    return w.float() * scale


@pytest.mark.parametrize("heads", [8, 16])
def test_split_kv_b_keeps_checkpoint_blocks(heads: int):
    nope, v, kv = 192, 256, 512
    rows = heads * (nope + v)
    w = (torch.randn(rows, kv) * 4).to(torch.float8_e4m3fn)
    s = torch.rand((rows + 127) // 128, kv // 128) + 0.5
    ref = _dequant(w, s, 128, 128).view(heads, nope + v, kv)

    w_uk, s_uk, w_uv, s_uv = split_kv_b(w, s, heads, nope, v)

    assert w_uk.shape == (heads * kv, nope) and s_uk.shape == (heads * kv // 128, 3)
    assert w_uv.shape == (heads * v, kv) and s_uv.shape == (heads * v // 64, 4)
    uk = _dequant(w_uk, s_uk, 128, 64).view(heads, kv, nope)
    uv = _dequant(w_uv, s_uv, 64, 128).view(heads, v, kv)
    torch.testing.assert_close(uk, ref[:, :nope].transpose(1, 2), rtol=0, atol=0)
    torch.testing.assert_close(uv, ref[:, nope:], rtol=0, atol=0)
