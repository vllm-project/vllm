# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The staged-weight GDN projection (`gdn_projection`, a CuTe DSL kernel)
against an FP32 reference at the GDN layer's TP=4 rank shapes: one and two
weight segments, row counts that leave MMA columns empty, weight segments whose
last 16-row unit is partial, stale shared memory, and CUDA graph replay."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(90),
    reason="the staged-weight projection requires SM90+",
)

if current_platform.is_cuda():
    from vllm.models.qwen4_exp.nvidia.specialized_decode.gdn.proj import (
        gdn_projection,
    )

HIDDEN = 2560
# TP=4 rank: in_proj_qkvz [4096, 2560], in_proj_ba [24, 2560] (a partial last
# unit), out_proj [2560, 1536].
SHAPES = {"in": ((4096, 24), HIDDEN), "out": ((2560,), 1536)}
ROWS = [1, 3, 4, 7, 8, 9, 12, 16]


def _weights(kind: str, seed: int) -> tuple[list[torch.Tensor], int]:
    ns, k = SHAPES[kind]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return [
        (torch.randn(n, k, generator=gen, device="cuda") * k**-0.5).bfloat16()
        for n in ns
    ], k


def _bound(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Within one BF16 rounding of the FP32 reference's largest value."""
    assert torch.isfinite(out).all()
    err = (out.float() - ref).abs().max().item()
    assert err <= 2**-7 * ref.abs().max().item(), (err, ref.abs().max().item())


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("kind", ["in", "out"])
def test_projection_matches_fp32(rows, kind):
    weights, k = _weights(kind, rows)
    gen = torch.Generator(device="cuda").manual_seed(100 + rows)
    x = torch.randn(rows, k, generator=gen, device="cuda").bfloat16()
    outs = gdn_projection(x, weights)
    for w, out in zip(weights, outs):
        _bound(out, x.float() @ w.float().T)


@pytest.mark.parametrize("rows", [4, 9])
def test_strided_activations_and_outputs(rows):
    """A column view of a wider activation buffer, outputs as column views of
    one buffer (as the core reads them)."""
    weights, k = _weights("in", 7)
    gen = torch.Generator(device="cuda").manual_seed(8)
    wide = torch.randn(rows, k + 64, generator=gen, device="cuda").bfloat16()
    x = wide[:, :k]
    buffer = torch.full((rows, 4096 + 24 + 8), float("nan"), device="cuda").bfloat16()
    outs = [buffer[:, :4096], buffer[:, 4096 : 4096 + 24]]
    gdn_projection(x, weights, outs)
    for w, out in zip(weights, outs):
        _bound(out, x.float() @ w.float().T)
    assert torch.isnan(buffer[:, 4096 + 24 :].float()).all(), "wrote past n"


def test_stale_shared_memory_does_not_reach_outputs():
    """A launch whose partial unit (ba's last 8 rows) reads shared memory a
    previous launch filled with NaN weights: the outputs stay finite."""
    weights, k = _weights("in", 3)
    x = torch.randn(4, k, device="cuda").bfloat16()
    poison = [torch.full_like(w, float("nan")) for w in weights]
    gdn_projection(x, poison)
    outs = gdn_projection(x, weights)
    for w, out in zip(weights, outs):
        _bound(out, x.float() @ w.float().T)


@pytest.mark.parametrize("rows", [4, 16])
def test_replays_bitwise_in_a_cuda_graph(rows):
    weights, k = _weights("in", 5)
    x = torch.randn(rows, k, device="cuda").bfloat16()
    eager = [o.clone() for o in gdn_projection(x, weights)]
    outs = [torch.empty_like(o) for o in eager]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gdn_projection(x, weights, outs)
    for _ in range(2):
        for o in outs:
            o.fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        for o, e in zip(outs, eager):
            assert torch.equal(o, e)
