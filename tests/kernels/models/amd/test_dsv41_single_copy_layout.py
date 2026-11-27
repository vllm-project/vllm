# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx942 mono copy of the routed experts (weights942) is the layout that
AITER's a16w4 Triton GEMM reads with w_layout="GFX942_TILE_MAJOR".

With monokernel_shared_experts, vLLM gives that GEMM the mono copy instead of
its own padded copy. This test checks that vLLM's packers and AITER's
gfx942_tile_major_pack give the same bytes and the same scale label, so the two
repositories cannot drift apart without a failing test.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("AITER needs ROCm", allow_module_level=True)

shuffle = pytest.importorskip("aiter.ops.triton.utils.shuffle")
if not hasattr(shuffle, "gfx942_tile_major_pack"):
    pytest.skip(
        "this AITER has no gfx942_tile_major_pack (ROCm/aiter#8454)",
        allow_module_level=True,
    )

from vllm.models.deepseek_v41.amd.mono.weights942 import (  # noqa: E402
    codes_tile_rounds,
    fp4_tile_major,
)

HIDDEN = 5120


@pytest.mark.parametrize("tp", [4, 8])
def test_mono_copy_is_aiter_tile_major(tp):
    """w13 (scales in rounds) and w2 (row-major scales) at the DeepSeek-V4.1
    shapes. TP4 w2 rows end in a 32-byte K step, TP8 w2 rows in a 16-byte one."""
    experts, inter = 4, 2304 // tp
    u8 = torch.uint8
    w13 = torch.randint(0, 256, (experts, 2 * inter, HIDDEN // 2), dtype=u8)
    s13 = torch.randint(0, 256, (experts, 2 * inter, HIDDEN // 32), dtype=u8)
    w2 = torch.randint(0, 256, (experts, HIDDEN, inter // 2), dtype=u8)
    s2 = torch.randint(0, 256, (experts, HIDDEN, inter // 32), dtype=u8)

    t13, t13_s, swz13 = shuffle.gfx942_tile_major_pack(w13, s13, scale_rounds=True)
    assert swz13 == "GFX942_TILE_ROUNDS"
    assert torch.equal(t13.transpose(1, 2), fp4_tile_major(w13))
    assert torch.equal(t13_s.transpose(1, 2), codes_tile_rounds(s13))

    t2, t2_s, swz2 = shuffle.gfx942_tile_major_pack(w2, s2, scale_rounds=False)
    assert swz2 is None
    assert torch.equal(t2.transpose(1, 2), fp4_tile_major(w2))
    assert torch.equal(t2_s.transpose(1, 2), s2)
