# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's shared expert GEMVs against PyTorch, on FP8 weights
with UE8M0 block scales in DeepGEMM's packed layout."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

HIDDEN, INTER, LIMIT = 4096, 512, 10.0
FP8 = torch.float8_e4m3fn

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="Requires SM100"
)


def _quant(x, ue8m0):
    """Dequantized FP8 per 128 with scale amax / 448 (a power of two when
    ``ue8m0``), like per_token_group_quant_fp8."""
    blocks = x.float().unflatten(-1, (-1, 128))
    amax = blocks.abs().amax(-1).clamp_min(1e-10)
    # IEEE division: dividing by a Python scalar multiplies by its reciprocal.
    scale = amax / torch.full_like(amax, 448.0)
    if ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (blocks * (1 / scale[..., None])).clamp(-448, 448).to(FP8)
    return (q.float() * scale[..., None]).flatten(-2)


def _weight(g, n, k):
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.5).to(FP8)
    exponents = torch.randint(
        118, 124, (n // 128, k // 128), generator=g, device="cuda"
    )
    return w, exponents.to(torch.int32)


def _packed(exponents, n):
    """DeepGEMM's int32 [n, k / 512] column-major packing of per-row UE8M0."""
    rows = exponents.repeat_interleave(128, 0)  # [n, k / 128]
    shifts = torch.arange(rows.shape[1], device=rows.device) % 4 * 8
    words = (rows << shifts).view(n, -1, 4).sum(-1, dtype=torch.int64).to(torch.int32)
    return words.T.contiguous().T


@pytest.mark.parametrize("ue8m0", [False, True])
@pytest.mark.parametrize("tokens", [1, 2])
def test_shared_expert_matches_reference(tokens, ue8m0):
    from vllm.models.glm5next.nvidia.specialized_decode.shared_expert import (
        SharedExpert,
    )

    g = torch.Generator(device="cuda").manual_seed(tokens)
    w13, e13 = _weight(g, 2 * INTER, HIDDEN)  # [gate; up]
    w2, e2 = _weight(g, HIDDEN, INTER)
    shared = SharedExpert(
        w13, _packed(e13, 2 * INTER), w2, _packed(e2, HIDDEN), LIMIT, ue8m0
    )
    x = torch.randn(tokens, HIDDEN, generator=g, device="cuda").bfloat16()
    x[:, :128] = 0  # an all-zero group takes the 1e-10 amax floor
    out = shared(x)

    def dequant(w, e):
        s = (e << 23).view(torch.float32)
        return w.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)

    gate_up = (_quant(x, ue8m0) @ dequant(w13, e13).T).bfloat16().float()
    gate, up = gate_up.split(INTER, dim=-1)
    act = (F.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)).bfloat16()
    ref = _quant(act, ue8m0) @ dequant(w2, e2).T
    torch.testing.assert_close(out.float(), ref, atol=5e-3 * ref.abs().max(), rtol=0)
